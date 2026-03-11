from configparser import ConfigParser
from nipype import DataGrabber, Node, Workflow, MapNode, Merge
from niflow.nipype1.workflows.dmri.fsl.dti import bedpostx_parallel
from diffusion_pipelines.workflows import init_preprocess_wf, init_recon_wf
from pathlib import Path
from nipype.interfaces.fsl import ProbTrackX2, BEDPOSTX5
import nipype.interfaces.ants as ants


def init_tracto_wf(
    config_file,
    name="tracto",
    n_fibres=3,
    fudge=1,
    burn_in=1000,
    n_jumps=1250,
    sample_every=25,
    output_dir=".",
):

    config = ConfigParser()
    config.read(config_file)

    roi_source = Node(DataGrabber(infields=[]), name="rois")
    roi_source.inputs.sort_filelist = True
    roi_source.inputs.base_directory = config["ROIS"]["directory"]
    roi_source.inputs.template = "*.nii.gz"

    # setup preprocess workflow
    preprocess = init_preprocess_wf(output_dir=output_dir)
    preprocess.inputs.input_subject.dwi = Path(
        config["SUBJECT"]["directory"], config["SUBJECT"]["dwi"]
    )
    preprocess.inputs.input_subject.bval = Path(
        config["SUBJECT"]["directory"], config["SUBJECT"]["bval"]
    )
    preprocess.inputs.input_subject.bvec = Path(
        config["SUBJECT"]["directory"], config["SUBJECT"]["bvec"]
    )
    preprocess.inputs.input_template.T1 = Path(
        config["TEMPLATE"]["directory"], config["TEMPLATE"]["T1"]
    )
    preprocess.inputs.input_template.T2 = Path(
        config["TEMPLATE"]["directory"], config["TEMPLATE"]["T2"]
    )
    preprocess.inputs.input_template.mask = Path(
        config["TEMPLATE"]["directory"], config["TEMPLATE"]["mask"]
    )

    # setup recon workflow
    recon = init_recon_wf(output_dir=output_dir)
    recon.inputs.input_subject.subject_id = config["SUBJECT"]["id"]
    recon.inputs.input_subject.subjects_dir = config["SUBJECT"]["directory"]
    recon.inputs.input_subject.T1 = Path(
        config["SUBJECT"]["directory"], config["SUBJECT"]["T1"]
    )
    recon.inputs.input_subject.dwi = Path(
        config["SUBJECT"]["directory"], config["SUBJECT"]["dwi"]
    )
    recon.inputs.input_subject.bval = Path(
        config["SUBJECT"]["directory"], config["SUBJECT"]["bval"]
    )
    recon.inputs.input_subject.bvec = Path(
        config["SUBJECT"]["directory"], config["SUBJECT"]["bvec"]
    )
    recon.inputs.input_template.T1 = Path(
        config["TEMPLATE"]["directory"], config["TEMPLATE"]["T1"]
    )
    recon.inputs.input_template.T2 = Path(
        config["TEMPLATE"]["directory"], config["TEMPLATE"]["T2"]
    )
    recon.inputs.input_template.mask = Path(
        config["TEMPLATE"]["directory"], config["TEMPLATE"]["mask"]
    )

    # node for applying registration from recon to ROIs
    apply_registration = MapNode(
        interface=ants.ApplyTransforms(),
        name="apply_registration",
        iterfield=["input_image"],
    )
    apply_registration.inputs.dimension = 3
    apply_registration.inputs.input_image_type = 3
    apply_registration.inputs.interpolation = "NearestNeighbor"

    # node for joining seeds
    join_seeds = Node(
        interface=Merge(2),
        name="join_seeds",
    )

    # bedpost_gpu
    bedpost_gpu = Node(interface=BEDPOSTX5(), name="bedpost_gpu")
    bedpost_gpu.inputs.n_fibres = n_fibres
    bedpost_gpu.inputs.fudge = fudge
    bedpost_gpu.inputs.burn_in = burn_in
    bedpost_gpu.inputs.n_jumps = n_jumps
    bedpost_gpu.inputs.sample_every = sample_every
    bedpost_gpu.inputs.use_gpu = True

    # setup probtrackx2 node
    pbx2 = Node(
        interface=ProbTrackX2(),
        name="probtrackx2",
    )
    pbx2.inputs.n_samples = 5000
    pbx2.inputs.n_steps = 2000
    pbx2.inputs.step_length = 0.5
    pbx2.inputs.omatrix1 = True
    pbx2.inputs.distthresh1 = 5
    pbx2.inputs.args = " --ompl --fibthresh=0.01 "

    workflow = Workflow(name=name, base_dir=output_dir)
    workflow.connect(
        [
            (roi_source, apply_registration, [("outfiles", "input_image")]),
            (
                recon,
                apply_registration,
                [
                    ("output.reg_nl_forward_transforms", "transforms"),
                    (
                        "output.reg_nl_forward_invert_flags",
                        "invert_transform_flags",
                    ),
                    ("output.mri_convert_reference_image", "reference_image"),
                ],
            ),
            (recon, join_seeds, [("output.shrunk_surface", "in1")]),
            (apply_registration, join_seeds, [("output_image", "in2")]),
            (
                join_seeds,
                pbx2,
                [
                    ("out", "seed"),
                ],
            ),
            (
                preprocess,
                bedpost_gpu,
                [
                    ("output.bval", "bvals"),
                    ("output.bvec_rotated", "bvecs"),
                    ("output.dwi_rigid_registered", "dwi"),
                    ("output.mask", "mask"),
                ],
            ),
            (
                preprocess,
                pbx2,
                [
                    ("output.mask", "mask"),
                ],
            ),
            (
                bedpost_gpu,
                pbx2,
                [
                    ("merged_thsamples", "thsamples"),
                    ("merged_fsamples", "fsamples"),
                    ("merged_phsamples", "phsamples"),
                ],
            ),
        ]
    )
    return workflow

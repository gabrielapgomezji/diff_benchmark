import inspect
from nipype import IdentityInterface, Node, Workflow, MapNode
import nipype.interfaces.fsl as fsl
import nipype.interfaces.ants as ants
import nipype.interfaces.freesurfer as fs
from niflow.nipype1.workflows.dmri.fsl.epi import create_eddy_correct_pipeline
from nipype.interfaces import utility
from nipype.interfaces.utility.wrappers import Function
from .report import init_report_wf
from .bids import init_bidsdata_wf
from .sink import init_sink_wf
from pathlib import Path
from niworkflows.anat.coregistration import init_bbreg_wf
from niworkflows.workflows.epi.refmap import init_epi_reference_wf
from niworkflows.interfaces.bids import BIDSFreeSurferDir
import os
from sdcflows.workflows.ancillary import init_brainextraction_wf

from nipype.interfaces.base import (
    TraitedSpec,
    CommandLineInputSpec,
    CommandLine,
    File,
)


class SynthStripInputSpec(CommandLineInputSpec):
    in_file = File(
        desc="Input image to skullstrip",
        exists=True,
        mandatory=True,
        argstr="--image %s",
    )
    out_file = File(
        desc="Output skullstripped image",
        argstr="--out %s",
        name_source="in_file",
        name_template="%s_stripped",
        keep_extension=True,
    )
    mask_file = File(
        desc="Output brain mask",
        argstr="--mask %s",
        name_source="in_file",
        name_template="%s_mask",
        keep_extension=True,
    )


class SynthStripOutputSpec(TraitedSpec):
    out_file = File(desc="Save stripped image to path.")
    mask_file = File(desc="Save brain mask to path.")


class SynthStrip(CommandLine):
    input_spec = SynthStripInputSpec
    output_spec = SynthStripOutputSpec
    _cmd = "python /opt/freesurfer/freesurfer/python/scripts/mri_synthstrip"

    def _list_outputs(self):
        outputs = self.output_spec().get()

        # Generate output file paths based on input file and templates
        if self.inputs.in_file:
            from nipype.utils.filemanip import split_filename

            # Get the input file path and split it
            in_file = self.inputs.in_file
            path, fname, ext = split_filename(in_file)

            # Generate output file paths using the name templates
            outputs["out_file"] = os.path.join(
                os.getcwd(), f"{fname}_stripped{ext}"
            )
            outputs["mask_file"] = os.path.join(
                os.getcwd(), f"{fname}_mask{ext}"
            )

            # If explicit output paths were provided in inputs
            # use those instead
            if hasattr(self.inputs, "out_file") and self.inputs.out_file:
                outputs["out_file"] = os.path.abspath(self.inputs.out_file)
            if hasattr(self.inputs, "mask_file") and self.inputs.mask_file:
                outputs["mask_file"] = os.path.abspath(self.inputs.mask_file)

        return outputs


class MPPCAInputSpec(CommandLineInputSpec):
    in_file = File(
        desc="Input image to skullstrip",
        exists=True,
        mandatory=True,
        argstr="%s",
    )
    out_file = File(
        desc="Output skullstripped image",
        argstr="--out_denoised %s",
        name_source="in_file",
        name_template="%s_mppca",
        keep_extension=True,
    )


class MPPCAOutputSpec(TraitedSpec):
    out_file = File(desc="Save MPPCA denoised image to path.")


class MPPCA(CommandLine):
    input_spec = MPPCAInputSpec
    output_spec = MPPCAOutputSpec
    _cmd = "dipy_denoise_mppca"

    def _list_outputs(self):
        outputs = self.output_spec().get()

        # Generate output file paths based on input file and templates
        if self.inputs.in_file:
            from nipype.utils.filemanip import split_filename

            # Get the input file path and split it
            in_file = self.inputs.in_file
            path, fname, ext = split_filename(in_file)

            # Generate output file paths using the name templates
            outputs["out_file"] = os.path.join(
                os.getcwd(), f"{fname}_mppca{ext}"
            )

            # If explicit output paths were provided in inputs
            # use those instead
            if hasattr(self.inputs, "out_file") and self.inputs.out_file:
                outputs["out_file"] = os.path.abspath(self.inputs.out_file)

        return outputs


class UnringInputSpec(CommandLineInputSpec):
    in_file = File(
        desc="Input image to unring",
        exists=True,
        mandatory=True,
        argstr="%s",
    )
    out_file = File(
        desc="Output Gibbs unringed image",
        argstr="--out_unring %s",
        name_source="in_file",
        name_template="%s_unring",
        keep_extension=True,
    )


class UnringOutputSpec(TraitedSpec):
    out_file = File(desc="Save Gibbs unringed denoised image to path.")


class Unring(CommandLine):
    input_spec = UnringInputSpec
    output_spec = UnringOutputSpec
    _cmd = "dipy_gibbs_ringing"

    def _list_outputs(self):
        outputs = self.output_spec().get()

        # Generate output file paths based on input file and templates
        if self.inputs.in_file:
            from nipype.utils.filemanip import split_filename

            # Get the input file path and split it
            in_file = self.inputs.in_file
            path, fname, ext = split_filename(in_file)

            # Generate output file paths using the name templates
            outputs["out_file"] = os.path.join(
                os.getcwd(), f"{fname}_unring{ext}"
            )

            # If explicit output paths were provided in inputs
            # use those instead
            if hasattr(self.inputs, "out_file") and self.inputs.out_file:
                outputs["out_file"] = os.path.abspath(self.inputs.out_file)

        return outputs


def _set_inputs_outputs(config, preproc_wf):
    # bids dataset
    bidsdata_wf = init_bidsdata_wf(config=config)
    # outputs
    sink_wf = init_sink_wf(config=config)
    # get freesurfer directory
    fsdir = Node(
        BIDSFreeSurferDir(
            derivatives=config.output_dir,
            freesurfer_home=os.getenv("FREESURFER_HOME"),
            spaces=config.output_spaces.get_fs_spaces(),
        ),
        name="fsdir_preproc",
    )
    # create the full workflow
    preproc_wf.connect(
        [
            (
                bidsdata_wf,
                preproc_wf.get_node("input_subject"),
                [
                    ("output.preprocessed_t1", "preprocessed_t1"),
                    (
                        "output.preprocessed_t1_mask",
                        "preprocessed_t1_mask",
                    ),
                    (
                        "output.fsnative2t1w_xfm",
                        "fsnative2t1w_xfm",
                    ),
                    ("output.dwi", "dwi"),
                    ("output.bval", "bval"),
                    ("output.bvec", "bvec"),
                    ("decode_entities.bids_entities", "bids_entities"),
                    (
                        "output.plot_recon_surface_on_t1",
                        "plot_recon_surface_on_t1",
                    ),
                    (
                        "output.plot_recon_segmentations_on_t1",
                        "plot_recon_segmentations_on_t1",
                    ),
                    ("output.ribbon_mask", "ribbon_mask"),
                ],
            ),
            (
                fsdir,
                preproc_wf.get_node("input_subject"),
                [("subjects_dir", "fs_subjects_dir")],
            ),
            (
                bidsdata_wf,
                sink_wf,
                [
                    (
                        "decode_entities.bids_entities",
                        "sinkinputnode.bids_entities",
                    )
                ],
            ),
            (
                preproc_wf.get_node("output"),
                sink_wf.get_node("sink"),
                [
                    (
                        "dwi_rigid_registered",
                        "diffusion_preprocess.@registered_dwi",
                    ),
                    ("eddy_corrected", "diffusion_preprocess.@eddy_corrected"),
                    ("mask", "diffusion_preprocess.@mask"),
                    (
                        "registered_mean_bzero",
                        "diffusion_preprocess.@registered_mean_bzero",
                    ),
                    ("bvec_rotated", "diffusion_preprocess.@bvec_rotated"),
                ],
            ),
            (
                preproc_wf.get_node("report"),
                sink_wf.get_node("sink"),
                [
                    (
                        "report_outputnode.out_file",
                        "diffusion_preprocess.@report",
                    )
                ],
            ),
        ]
    )
    return preproc_wf


def _preprocess_wf(config, name="diffusion_preprocess", output_dir="."):

    def _get_zero_indexes(bval, bval_threshold):
        """Get the indexes of the b=0 volumes."""
        import numpy as np

        bvals = np.loadtxt(bval)
        zero_indexes = bvals <= bval_threshold
        return zero_indexes.tolist()

    GetZeroIndexes = Function(
        input_names=["bval", "bval_threshold"],
        output_names=["zero_indexes"],
        function=_get_zero_indexes,
    )

    get_initial_zero_indexes = Node(
        GetZeroIndexes, name="get_initial_zero_indexes"
    )
    get_initial_zero_indexes.inputs.bval_threshold = config.b0_threshold

    get_eddy_zero_indexes = get_initial_zero_indexes.clone(
        "get_eddy_zero_indexes"
    )
    get_eddy_zero_indexes.inputs.bval_threshold = config.b0_threshold

    get_registered_zero_indexes = get_initial_zero_indexes.clone(
        "get_registered_zero_indexes"
    )
    get_registered_zero_indexes.inputs.bval_threshold = config.b0_threshold

    get_initial_mean_bzero = init_epi_reference_wf(
        name="get_initial_mean_bzero",
        omp_nthreads=config.omp_nthreads,
        auto_bold_nss=False,
    )

    get_mppca_mean_bzero = init_epi_reference_wf(
        name="get_mppca_mean_bzero",
        omp_nthreads=config.omp_nthreads,
        auto_bold_nss=False,
    )

    get_gibbs_unringed_mean_bzero = init_epi_reference_wf(
        name="get_gibbs_unringed_mean_bzero",
        omp_nthreads=config.omp_nthreads,
        auto_bold_nss=False,
    )

    get_eddy_mean_bzero = init_epi_reference_wf(
        name="get_eddy_mean_bzero",
        omp_nthreads=config.omp_nthreads,
        auto_bold_nss=False,
    )
    get_registered_mean_bzero = init_epi_reference_wf(
        name="get_registered_mean_bzero",
        omp_nthreads=config.omp_nthreads,
        auto_bold_nss=False,
    )

    def get_subject_id(bids_entities):
        """Get the subject id from the BIDS entities."""
        return f"sub-{bids_entities['subject']}"

    GetSubjectID = Function(
        input_names=["bids_entities"],
        output_names=["subject_id"],
        function=get_subject_id,
    )
    get_subject_id_node = Node(GetSubjectID, name="get_subject_id")

    def rotate_gradients_lta(lta_file, gradient_file):
        import os
        import os.path
        import numpy as np
        from scipy.linalg import polar

        # Parse the LTA file to extract the transformation matrix
        with open(lta_file, "r") as f:
            lines = f.readlines()

        # Find the matrix section (after "1 4 4" line)
        matrix_start = None
        for i, line in enumerate(lines):
            if line.strip() == "1 4 4":
                matrix_start = i + 1
                break

        if matrix_start is None:
            raise ValueError(
                "Could not find transformation matrix in LTA file"
            )

        # Read the 4x4 transformation matrix
        matrix_lines = lines[matrix_start : matrix_start + 4]
        affine = np.array(
            [list(map(float, line.strip().split())) for line in matrix_lines]
        )

        # Extract rotation component using polar decomposition
        u, p = polar(affine[:3, :3], side="right")

        # Load and rotate gradients
        gradients = np.loadtxt(gradient_file)
        new_gradients = np.linalg.solve(u, gradients).T

        # Save rotated gradients
        name, ext = os.path.splitext(os.path.basename(gradient_file))
        output_name = os.path.join(os.getcwd(), f"{name}_rot{ext}")
        np.savetxt(output_name, new_gradients)

        return output_name

    RotateGradientsLTA = Function(
        input_names=["lta_file", "gradient_file"],
        output_names=["rotated_gradients"],
        function=rotate_gradients_lta,
    )

    input_subject = Node(
        IdentityInterface(
            fields=[
                "preprocessed_t1",
                "preprocessed_t1_mask",
                "fsnative2t1w_xfm",
                "fs_subjects_dir",
                "dwi",
                "bval",
                "bvec",
                "bids_entities",
                "plot_recon_surface_on_t1",
                "plot_recon_segmentations_on_t1",
                "ribbon_mask",
            ],
        ),
        name="input_subject",
    )
    input_template = Node(
        IdentityInterface(
            fields=["T2", "mask"],
        ),
        name="input_template",
    )

    output = Node(
        IdentityInterface(
            fields=[
                "dwi_rigid_registered",
                "bval",
                "bvec_rotated",
                "mask",
                "ribbon_mask",
                "rigid_dwi_2_t1",
                "eddy_corrected",
                "dwi_initial",
                "dwi_masked",
                "bet_mask",
                "t1_initial",
                "t1_masked",
                "bids_entities",
                "plot_recon_surface_on_t1",
                "plot_recon_segmentations_on_t1",
                "initial_mean_bzero",
                "eddy_mean_bzero",
                "registered_mean_bzero",
                "gibbs_unringed_denoised",
                "mppca_denoised",
            ]
        ),
        name="output",
    )

    fslroi = Node(interface=fsl.ExtractROI(), name="fslroi")
    fslroi.inputs.t_min = 0
    fslroi.inputs.t_size = 1

    strip_mean_bzero = Node(interface=fsl.ApplyMask(), name="strip_mean_bzero")

    strip_dwi = Node(interface=fsl.ApplyMask(), name="strip_dwi")

    strip_t1 = Node(interface=fsl.ApplyMask(), name="strip_t1")

    synthstrip = Node(interface=SynthStrip(), name="synthstrip")

    mppca = Node(interface=MPPCA(), name="mppca")

    gibbs_unring = Node(interface=Unring(), name="gibbs_unring")

    eddycorrect = create_eddy_correct_pipeline("eddycorrect")
    eddycorrect.inputs.inputnode.ref_num = 0

    rotate_gradients = Node(
        interface=RotateGradientsLTA, name="rotate_gradients"
    )

    apply_registration = Node(
        interface=fs.ApplyVolTransform(), name="apply_registration"
    )

    apply_registration_mask = Node(
        interface=fs.ApplyVolTransform(), name="apply_registration_mask"
    )

    # Add renaming node for registered mean bzero
    def rename_registered_mean_bzero(in_file):
        import os
        import shutil

        output_name = os.path.join(os.getcwd(), "registered_mean_bzero.nii.gz")
        shutil.copy(in_file, output_name)
        return output_name

    RenameRegisteredMeanBzero = Function(
        input_names=["in_file"],
        output_names=["out_file"],
        function=rename_registered_mean_bzero,
    )
    rename_registered_bzero = Node(
        RenameRegisteredMeanBzero, name="rename_registered_bzero"
    )

    bbreg_wf = init_bbreg_wf(
        name="bbreg_wf",
        omp_nthreads=config.omp_nthreads,
        use_bbr=True,
        epi2t1w_dof=12,
    )

    report = init_report_wf(
        name="report", calling_wf_name=name, output_dir=output_dir
    )

    workflow = Workflow(name=name, base_dir=output_dir)
    workflow.connect(
        [
            (input_subject, get_initial_zero_indexes, [("bval", "bval")]),
            # get mean of b=0 volumes of the input dwi file
            (
                input_subject,
                get_initial_mean_bzero,
                [("dwi", "inputnode.in_files")],
            ),
            (
                get_initial_zero_indexes,
                get_initial_mean_bzero,
                [("zero_indexes", "inputnode.t_masks")],
            ),
            # get mask from the mean b=0 volumes
            (
                get_initial_mean_bzero,
                synthstrip,
                [("outputnode.epi_ref_file", "in_file")],
            ),
            # apply mask to mean b=0 output
            (
                get_initial_mean_bzero,
                strip_mean_bzero,
                [("outputnode.epi_ref_file", "in_file")],
            ),
            (
                synthstrip,
                strip_mean_bzero,
                [("mask_file", "mask_file")],
            ),
            # apply the mask to the dwi
            (input_subject, strip_dwi, [("dwi", "in_file")]),
            (
                synthstrip,
                strip_dwi,
                [("mask_file", "mask_file")],
            ),
            # apply mask to the preprocessed subject T1
            (input_subject, strip_t1, [("preprocessed_t1", "in_file")]),
            (input_subject, strip_t1, [("preprocessed_t1_mask", "mask_file")]),
            # denoise the skull-stripped dwi with MP-PCA
            (strip_dwi, mppca, [("out_file", "in_file")]),
            # Gibbs unring the denoised dwi
            (mppca, gibbs_unring, [("out_file", "in_file")]),
            # eddycorrect the unringed dwi
            (gibbs_unring, eddycorrect, [("out_file", "inputnode.in_file")]),
            # compute the mean of the b=0 eddycorrected volumes
            (
                eddycorrect,
                get_eddy_mean_bzero,
                [("outputnode.eddy_corrected", "inputnode.in_files")],
            ),
            (
                get_initial_zero_indexes,
                get_eddy_mean_bzero,
                [("zero_indexes", "inputnode.t_masks")],
            ),
            # register the skull-stripped dwi to the skull-stripped subject T1
            (
                get_eddy_mean_bzero,
                bbreg_wf,
                [("outputnode.epi_ref_file", "inputnode.in_file")],
            ),
            (
                input_subject,
                bbreg_wf,
                [("fsnative2t1w_xfm", "inputnode.fsnative2t1w_xfm")],
            ),
            (
                input_subject,
                get_subject_id_node,
                [("bids_entities", "bids_entities")],
            ),
            (
                get_subject_id_node,
                bbreg_wf,
                [("subject_id", "inputnode.subject_id")],
            ),
            (
                input_subject,
                bbreg_wf,
                [("fs_subjects_dir", "inputnode.subjects_dir")],
            ),
            # some matrix format conversions
            # rotate the gradients using the LTA file directly
            (input_subject, rotate_gradients, [("bvec", "gradient_file")]),
            (
                bbreg_wf.get_node("bbregister"),
                rotate_gradients,
                [("out_lta_file", "lta_file")],
            ),
            # apply the registration to the skull-stripped and eddy-corrected
            # dwi using FreeSurfer's ApplyVolTransform
            (
                bbreg_wf.get_node("bbregister"),
                apply_registration,
                [("out_lta_file", "lta_file")],
            ),
            (
                eddycorrect,
                apply_registration,
                [("outputnode.eddy_corrected", "source_file")],
            ),
            (
                strip_t1,
                apply_registration,
                [("out_file", "target_file")],
            ),
            # get a mean b=0 image of the registered dwi
            (
                apply_registration,
                get_registered_mean_bzero,
                [("transformed_file", "inputnode.in_files")],
            ),
            (
                get_initial_zero_indexes,
                get_registered_mean_bzero,
                [("zero_indexes", "inputnode.t_masks")],
            ),
            # rename the registered mean bzero for proper sink substitution
            (
                get_registered_mean_bzero,
                rename_registered_bzero,
                [("outputnode.epi_ref_file", "in_file")],
            ),
            # also apply the registration to the mask
            (
                bbreg_wf.get_node("bbregister"),
                apply_registration_mask,
                [("out_lta_file", "lta_file")],
            ),
            (
                synthstrip,
                apply_registration_mask,
                [("mask_file", "source_file")],
            ),
            (
                strip_t1,
                apply_registration_mask,
                [("out_file", "target_file")],
            ),
            # collect all the outputs in the output node
            # get subject id
            (input_subject, output, [("bids_entities", "bids_entities")]),
            # get the plots from smriprep
            (
                input_subject,
                output,
                [("plot_recon_surface_on_t1", "plot_recon_surface_on_t1")],
            ),
            (
                input_subject,
                output,
                [
                    (
                        "plot_recon_segmentations_on_t1",
                        "plot_recon_segmentations_on_t1",
                    )
                ],
            ),
            (strip_dwi, output, [("out_file", "dwi_masked")]),
            (
                bbreg_wf.get_node("bbregister"),
                output,
                [("out_lta_file", "rigid_dwi_2_t1")],
            ),
            (input_subject, output, [("preprocessed_t1", "t1_initial")]),
            (strip_t1, output, [("out_file", "t1_masked")]),
            (
                apply_registration,
                output,
                [("transformed_file", "dwi_rigid_registered")],
            ),
            (
                rotate_gradients,
                output,
                [("rotated_gradients", "bvec_rotated")],
            ),
            (input_subject, output, [("bval", "bval")]),
            (
                synthstrip,
                output,
                [("mask_file", "bet_mask")],
            ),
            (apply_registration_mask, output, [("transformed_file", "mask")]),
            (
                eddycorrect,
                output,
                [("outputnode.eddy_corrected", "eddy_corrected")],
            ),
            (input_subject, output, [("dwi", "dwi_initial")]),
            (input_subject, output, [("ribbon_mask", "ribbon_mask")]),
            (
                rename_registered_bzero,
                output,
                [("out_file", "registered_mean_bzero")],
            ),
            (
                get_initial_mean_bzero,
                output,
                [("outputnode.epi_ref_file", "initial_mean_bzero")],
            ),
            (
                get_eddy_mean_bzero,
                output,
                [("outputnode.epi_ref_file", "eddy_mean_bzero")],
            ),
            (
                mppca,
                get_mppca_mean_bzero,
                [("out_file", "inputnode.in_files")],
            ),
            (
                get_initial_zero_indexes,
                get_mppca_mean_bzero,
                [("zero_indexes", "inputnode.t_masks")],
            ),
            (
                gibbs_unring,
                get_gibbs_unringed_mean_bzero,
                [("out_file", "inputnode.in_files")],
            ),
            (
                get_initial_zero_indexes,
                get_gibbs_unringed_mean_bzero,
                [("zero_indexes", "inputnode.t_masks")],
            ),
            (
                get_mppca_mean_bzero,
                output,
                [("outputnode.epi_ref_file", "mppca_denoised")],
            ),
            (
                get_gibbs_unringed_mean_bzero,
                output,
                [("outputnode.epi_ref_file", "gibbs_unringed_denoised")],
            ),
            # connect the report workflow
            (
                output,
                report,
                [
                    ("dwi_initial", "report_inputnode.dwi_initial"),
                    ("dwi_masked", "report_inputnode.dwi_masked"),
                    ("bval", "report_inputnode.bval"),
                    (
                        "t1_initial",
                        "report_inputnode.t1_initial",
                    ),
                    (
                        "t1_masked",
                        "report_inputnode.t1_masked",
                    ),
                    ("eddy_corrected", "report_inputnode.eddy_corrected"),
                    ("mask", "report_inputnode.mask"),
                    ("bet_mask", "report_inputnode.bet_mask"),
                    (
                        "dwi_rigid_registered",
                        "report_inputnode.dwi_rigid_registered",
                    ),
                    ("bids_entities", "report_inputnode.bids_entities"),
                    (
                        "plot_recon_surface_on_t1",
                        "report_inputnode.plot_recon_surface_on_t1",
                    ),
                    (
                        "plot_recon_segmentations_on_t1",
                        "report_inputnode.plot_recon_segmentations_on_t1",
                    ),
                    (
                        "initial_mean_bzero",
                        "report_inputnode.initial_mean_bzero",
                    ),
                    ("eddy_mean_bzero", "report_inputnode.eddy_mean_bzero"),
                    (
                        "registered_mean_bzero",
                        "report_inputnode.registered_mean_bzero",
                    ),
                    ("ribbon_mask", "report_inputnode.ribbon_mask"),
                    (
                        "mppca_denoised",
                        "report_inputnode.mppca_denoised",
                    ),
                    (
                        "gibbs_unringed_denoised",
                        "report_inputnode.gibbs_unringed_denoised",
                    ),
                ],
            ),
        ]
    )

    return workflow


def init_preprocess_wf(output_dir=".", config=None):
    wf = _preprocess_wf(output_dir=output_dir, config=config)
    wf = _set_inputs_outputs(config, wf)
    return wf

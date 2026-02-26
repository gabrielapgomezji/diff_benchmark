# mostly a copy of get_parser function in https://github.com/nipreps/smriprep/blob/master/src/smriprep/cli/run.py


def get_parser():
    """Build parser object."""
    from argparse import ArgumentParser, RawTextHelpFormatter
    from pathlib import Path

    from niworkflows.utils.spaces import (
        OutputReferencesAction,
        Reference,
        SpatialReferences,
    )

    import smriprep

    def _drop_ses(value):
        return value.removeprefix("ses-")

    def _drop_sub(value):
        return value.removeprefix("sub-")

    parser = ArgumentParser(
        description="diffusion_pipelines",
        formatter_class=RawTextHelpFormatter,
    )

    # Arguments as specified by BIDS-Apps
    # required, positional arguments
    # IMPORTANT: they must go directly with the parser object
    parser.add_argument(
        "bids_dir",
        action="store",
        type=Path,
        help="the root folder of a BIDS valid dataset (sub-XXXXX folders should "
        "be found at the top level in this folder).",
    )
    parser.add_argument(
        "output_dir",
        action="store",
        type=Path,
        help="the output path for the outcomes of preprocessing and visual reports",
    )

    # optional arguments
    g_bids = parser.add_argument_group("Options for filtering BIDS queries")
    g_bids.add_argument(
        "--participant-label",
        "--participant_label",
        action="store",
        nargs="+",
        type=_drop_sub,
        help="a space delimited list of participant identifiers or a single "
        "identifier (the sub- prefix can be removed)",
    )
    g_bids.add_argument(
        "--session-label",
        nargs="+",
        type=_drop_ses,
        help="A space delimited list of session identifiers or a single "
        "identifier (the ses- prefix can be removed)",
    )
    g_bids.add_argument(
        "-d",
        "--derivatives",
        action="store",
        metavar="PATH",
        type=Path,
        nargs="*",
        help="Search PATH(s) for pre-computed derivatives.",
    )
    g_bids.add_argument(
        "--subject-anatomical-reference",
        choices=["first-lex", "unbiased", "sessionwise"],
        default="first",
        help="Method to produce the reference anatomical space:\n"
        '\t"first-lex" will use the first image in lexicographical order\n'
        '\t"unbiased" will construct an unbiased template from all images '
        '(previously "--longitudinal")\n'
        '\t"sessionwise" will independently process each session. If multiple runs are '
        'found, the behavior will be similar to "first-lex"',
    )
    g_bids.add_argument(
        "--bids-filter-file",
        action="store",
        type=Path,
        metavar="PATH",
        help="a JSON file describing custom BIDS input filters using pybids "
        "{<suffix>:{<entity>:<filter>,...},...} "
        "(https://github.com/bids-standard/pybids/blob/master/bids/layout/config/bids.json)",
    )

    g_perfm = parser.add_argument_group("Options to handle performance")
    g_perfm.add_argument(
        "--nprocs",
        "--ncpus",
        "--nthreads",
        "--n_cpus",
        "-n-cpus",
        action="store",
        type=int,
        help="number of CPUs to be used.",
    )
    g_perfm.add_argument(
        "--omp-nthreads",
        action="store",
        type=int,
        default=0,
        help="maximum number of threads per-process",
    )
    g_perfm.add_argument(
        "--mem-gb",
        "--mem_gb",
        action="store",
        default=0,
        type=float,
        help="upper bound memory limit for sMRIPrep processes (in GB).",
    )
    g_perfm.add_argument(
        "--low-mem",
        action="store_true",
        help="attempt to reduce memory usage (will increase disk usage in working directory)",
    )

    g_conf = parser.add_argument_group("Workflow configuration")
    g_conf.add_argument(
        "--output-spaces",
        nargs="*",
        action=OutputReferencesAction,
        default=SpatialReferences(),
        help="paths or keywords prescribing output spaces - "
        "standard spaces will be extracted for spatial normalization.",
    )
    g_conf.add_argument(
        "--longitudinal",
        action="store_false",
        help='DEPRECATED: use "--subject-anatomical-reference unbiased" instead',
    )
    g_conf.add_argument(
        "--b0-threshold",
        action="store",
        default=0,
        type=int,
        help="Threshold for B0 images - images with lower or equal values "
        "will be considered as B0 images. Default is 0",
    )

    #  ANTs options
    g_ants = parser.add_argument_group(
        "Specific options for ANTs registrations"
    )
    g_ants.add_argument(
        "--skull-strip-template",
        default="OASIS30ANTs",
        type=Reference.from_string,
        help="select a template for skull-stripping with antsBrainExtraction",
    )
    g_ants.add_argument(
        "--skull-strip-fixed-seed",
        action="store_true",
        help="do not use a random seed for skull-stripping - will ensure "
        "run-to-run replicability when used with --omp-nthreads 1",
    )
    g_ants.add_argument(
        "--skull-strip-mode",
        action="store",
        choices=("auto", "skip", "force"),
        default="auto",
        help="determiner for T1-weighted skull stripping (force ensures skull "
        "stripping, skip ignores skull stripping, and auto automatically "
        "ignores skull stripping if pre-stripped brains are detected).",
    )

    # FreeSurfer options
    g_fs = parser.add_argument_group(
        "Specific options for FreeSurfer preprocessing"
    )
    g_fs.add_argument(
        "--fs-subjects-dir",
        metavar="PATH",
        type=Path,
        help="Path to existing FreeSurfer subjects directory to reuse. "
        "(default: OUTPUT_DIR/freesurfer)",
    )
    g_fs.add_argument(
        "--fs-no-resume",
        action="store_true",
        dest="fs_no_resume",
        help="EXPERT: Import pre-computed FreeSurfer reconstruction without resuming. "
        "The user is responsible for ensuring that all necessary files are present.",
    )
    g_fs.add_argument(
        "--cifti-output",
        nargs="?",
        const="91k",
        default=False,
        choices=("91k", "170k"),
        type=str,
        help="Output morphometry as CIFTI dense scalars. "
        "Optionally, the number of grayordinate can be specified "
        "(default is 91k, which equates to 2mm resolution)",
    )

    # Surface generation xor
    g_surfs = parser.add_argument_group("Surface preprocessing options")
    g_surfs.add_argument(
        "--no-submm-recon",
        action="store_false",
        dest="hires",
        help="disable sub-millimeter (hires) reconstruction",
    )
    g_surfs.add_argument(
        "--no-msm",
        action="store_false",
        dest="msm_sulc",
        help="Disable Multimodal Surface Matching surface registration.",
    )
    g_surfs_xor = g_surfs.add_mutually_exclusive_group()

    g_surfs_xor.add_argument(
        "--fs-no-reconall",
        action="store_false",
        dest="run_reconall",
        help="disable FreeSurfer surface preprocessing.",
    )

    g_other = parser.add_argument_group("Other options")
    g_other.add_argument(
        "-w",
        "--work-dir",
        action="store",
        type=Path,
        default=Path("work"),
        help="path where intermediate results should be stored",
    )
    g_other.add_argument(
        "--run-uuid",
        action="store",
        default=None,
        help="Specify UUID of previous run, to include error logs in report. "
        "No effect without --reports-only.",
    )
    g_other.add_argument(
        "--write-graph",
        action="store_true",
        default=False,
        help="Write workflow graph.",
    )
    g_other.add_argument(
        "--sloppy",
        action="store_true",
        default=False,
        help="Use low-quality tools for speed - TESTING ONLY",
    )
    g_other.add_argument(
        "--recon",
        action="store_true",
        default=False,
        help="Run the reconstruction wf (smriprep)",
    )
    g_other.add_argument(
        "--preproc",
        action="store_true",
        default=False,
        help="Preprocess the diffusion image.",
    )
    g_other.add_argument(
        "-pt1",
        "--preproc-t1",
        action="store",
        type=Path,
        default=None,
        help="path to the preprocessed (via reconstruction wf) T1-weighted image",
    )
    g_other.add_argument(
        "-pt1m",
        "--preproc-t1-mask",
        action="store",
        type=Path,
        default=None,
        help="path to the preprocessed (via reconstruction wf) T1-weighted mask",
    )
    g_other.add_argument(
        "-fs2t1",
        "--fs-native-to-t1w-xfm",
        action="store",
        type=Path,
        default=None,
        help="path to the fsnative2t1w_xfm transformation file",
    )
    g_other.add_argument(
        "--tracto",
        action="store_true",
        default=False,
        help="Run tractography pipeline.",
    )
    g_other.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run in debug mode.",
    )

    return parser

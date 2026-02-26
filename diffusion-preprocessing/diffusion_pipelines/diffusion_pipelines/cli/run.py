import os
import time
from diffusion_pipelines.workflows import (
    init_preprocess_wf,
    init_recon_wf,
    init_tracto_wf,
)

from nipype import config as nipype_config

from diffusion_pipelines.cli.arg_parser import get_parser


def _parse_pipelines(config):
    """
    Setup the pipeline based on the input arguments.
    """

    # if no pipelines are specified, we will only run reconstruction by default
    if config.recon and not config.preproc and not config.tracto:
        to_run = ["reconstruction"]

    # make sure the pipelines are in the correct order
    # if preprocessing and reconstruction is True in config,
    # run reconstruction first and then preprocessing
    if config.recon and config.preproc:
        to_run = ["reconstruction", "preprocessing"]
    # if only preprocessing is True in config, run preprocessing
    elif config.preproc and not config.recon:
        to_run = ["preprocessing"]

    # if tractography is True in config, it will run reconstruction and
    # preprocessing as well, so we just run tractography
    if config.tracto:
        to_run = ["tractography"]

    return to_run


def _run_pipeline(config, to_run):
    """
    Run the pipeline based on the config file.
    """

    # Create a timestamp in YYYYMMDD_HHMMSS format
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if config.run_uuid is None:
        config.run_uuid = f"{timestamp}_{config.participant_label[0]}"

    # pipeline to initialization function mapping
    pipeline_function = {
        "preprocessing": init_preprocess_wf,
        "reconstruction": init_recon_wf,
        "tractography": init_tracto_wf,
    }
    if config.debug:
        nipype_config.enable_debug_mode()
    for pipeline in to_run:
        # create the pipeline
        wf = pipeline_function[pipeline](
            output_dir=os.path.join(
                config.work_dir, f"{pipeline}_output_{config.run_uuid}"
            ),
            config=config,
        )
        wf.write_graph(
            graph2use="flat",
            dotfilename=os.path.join(
                config.work_dir,
                f"{pipeline}_output_{config.run_uuid}",
                "graph.dot",
            ),
            format="svg",
        )
        if config.nprocs > 1:
            wf.run(
                plugin="MultiProc",
                plugin_args={"n_procs": config.nprocs},
            )
        else:
            wf.run()


def main():
    """
    Main function to run the diffusion preprocessing pipeline.
    """
    config = get_parser().parse_args()
    _run_pipeline(config, _parse_pipelines(config))

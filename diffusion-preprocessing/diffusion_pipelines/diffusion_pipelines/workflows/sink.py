from nipype import IdentityInterface, Node, Workflow
from nipype.interfaces.utility import Function
from nipype.interfaces.io import DataSink


def init_sink_wf(config, name="sink_wf"):

    inputnode = Node(
        IdentityInterface(fields=["bids_entities"]), name="sinkinputnode"
    )

    ### build the full file name
    def build_substitutions(bids_entities):

        import os
        from pathlib import Path

        def _build_bids(bids_entities):
            replacements = {
                "subject": "sub-",
                "session": "_ses-",
                "acquisition": "_acq-",
                "direction": "_dir-",
                "part": "_part-",
            }
            bids_name = ""
            for key, value in bids_entities.items():
                if key in replacements:
                    bids_name += f"{replacements[key]}{value}"
            return bids_name

        bids_name = _build_bids(bids_entities)

        substitutions = [
            (
                "clipped_mask",
                f"{bids_name}_space-individualT1_desc-mask+bbreg_dwi",
            ),
            (
                "vol0000_flirt_merged_warped",
                f"{bids_name}_space-individualT1_desc-mppcadenoised+gibbsunringed+eddycorrected+bbreg_dwi",
            ),
            (
                "vol0000_flirt_merged",
                f"{bids_name}_desc-mppcadenoised+gibbsunringed+eddycorrected_dwi",
            ),
            (
                f"registered_mean_bzero",
                f"{bids_name}_space-individualT1_"
                "desc-mppcadenoised+gibbsunringed+eddycorrected+bbreg+meanb0_dwi",
            ),
            (
                f"{bids_name}_dwi_rot.bvec",
                f"{bids_name}_desc-rotated_dwi.bvec",
            ),
            (
                f"{bids_name}_report.html",
                f"{bids_name}_report.html",
            ),
        ]

        # add root directory in substitutions
        for i, (src, dst) in enumerate(substitutions):

            modality = dst.split("_")[-1].split(".")[0]

            if bids_entities.get("session"):
                prefix = os.path.join(
                    "sub-" + bids_entities["subject"],
                    "ses-" + bids_entities["session"],
                    modality,
                )
            else:
                prefix = os.path.join(
                    "sub-" + bids_entities["subject"], modality
                )

            substitutions[i] = (src, os.path.join(prefix, dst))

        return substitutions

    BuildSubstitutions = Function(
        input_names=["bids_entities"],
        output_names=["substitutions"],
        function=build_substitutions,
    )
    build_substitutions = Node(BuildSubstitutions, name="build_substitutions")

    ### DataSink node
    sink = Node(DataSink(), name="sink")
    sink.inputs.base_directory = str(config.output_dir)

    # Create the workflow
    sink_wf = Workflow(name=name)
    sink_wf.connect(
        [
            (
                inputnode,
                build_substitutions,
                [("bids_entities", "bids_entities")],
            ),
            (build_substitutions, sink, [("substitutions", "substitutions")]),
        ]
    )
    return sink_wf

from niworkflows.interfaces.reportlets.masks import SimpleShowMaskRPT
from niworkflows.interfaces.reportlets.registration import (
    SimpleBeforeAfterRPT as SimpleBeforeAfter,
)
from nipype.interfaces.utility.wrappers import Function
from nipype import IdentityInterface, Node, Workflow, Merge
import os

TEMPLATE_ROOT = os.path.join(os.path.dirname(__file__), "report_template")
REPORT_TEMPLATE = os.path.join(TEMPLATE_ROOT, "report_template.html")


def create_html_report(
    calling_wf_name,
    report_wf_name,
    template_path,
    output_dir,
    bids_entities,
    plots,
):
    import os
    import string
    from nilearn.plotting.html_document import HTMLDocument
    import base64

    def _embed_svg(to_embed, template_path=template_path):
        with open(template_path) as f:
            template_text = f.read()
        string_template = string.Template(template_text)
        string_text = string_template.safe_substitute(**to_embed)
        f.close()

        return string_text

    def _get_html_text(subject_id, *args):
        to_embed = {"subject_id": subject_id}
        recon_plots = {
            "T1w.svg": "plot_recon_surface_on_t1",
            "dseg.svg": "plot_recon_segmentations_on_t1",
        }
        for plot in args:
            if plot is not None:
                with open(plot, "r", encoding="utf-8") as f:
                    svg_text = f.read()
                f.close()
                # get the plot name from the path
                if "smriprep" in plot:
                    suffix = plot.split(os.path.sep)[-1].split("_")[-1]
                    plot_name = recon_plots[suffix]
                else:
                    plot_name = plot.split(os.path.sep)[-2]
                to_embed[plot_name] = svg_text
        return _embed_svg(to_embed)

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

    html_text = _get_html_text(bids_entities["subject"], *plots)
    bids_name = _build_bids(bids_entities)
    out_file = os.path.join(
        output_dir,
        calling_wf_name,
        report_wf_name,
        f"{bids_name}_report.html",
    )
    report_html = HTMLDocument(html_text).save_as_html(out_file)
    print(f"Report for {calling_wf_name} created at {out_file}")
    return out_file


def init_report_wf(calling_wf_name, output_dir, name="report"):
    """Create a workflow to generate a report for the diffusion preprocessing
    pipeline.

    Parameters
    ----------
    name : str, optional, by default "report"
        Name of the workflow
    output_dir : str, optional, by default "."
        Base directory to store the reports. The workflow will create a
        subdirectory called 'report' in this directory to store the reports.

    Returns
    -------
    workflow : nipype Workflow
        A nipype Workflow to generate the report for the diffusion
        preprocessing pipeline.
    """
    inputnode = Node(
        IdentityInterface(
            fields=[
                "dwi_initial",
                "dwi_masked",
                "bval",
                "eddy_corrected",
                "mask",
                "bet_mask",
                "dwi_rigid_registered",
                "t1_initial",
                "t1_masked",
                "bids_entities",
                "plot_recon_surface_on_t1",
                "plot_recon_segmentations_on_t1",
                "initial_mean_bzero",
                "eddy_mean_bzero",
                "registered_mean_bzero",
                "ribbon_mask",
                "mppca_denoised",
                "gibbs_unringed_denoised",
            ]
        ),
        name="report_inputnode",
    )
    outputnode = Node(
        IdentityInterface(fields=["out_file"]),
        name="report_outputnode",
    )
    # MPPCA
    plot_before_after_mppca = Node(
        SimpleBeforeAfter(), name="plot_before_after_mppca"
    )
    plot_before_after_mppca.inputs.before_label = "Before MP-PCA (Initial DWI)"
    plot_before_after_mppca.inputs.after_label = "After MP-PCA"

    # Gibbs Unringing
    plot_before_after_gibbs = Node(
        SimpleBeforeAfter(), name="plot_before_after_gibbs"
    )
    plot_before_after_gibbs.inputs.before_label = (
        "Before Gibbs Unringing (MP-PCA Denoised)"
    )
    plot_before_after_gibbs.inputs.after_label = "After Gibbs Unringing"

    # this node plots the before and after images of the eddy correction
    plot_before_after_eddy = Node(
        SimpleBeforeAfter(), name="plot_before_after_eddy"
    )
    # set labels for the before and after images
    plot_before_after_eddy.inputs.before_label = (
        "Before Eddy Correction (Gibbs Unringed + MP-PCA Denoised)"
    )
    plot_before_after_eddy.inputs.after_label = "Eddy Corrected DWI"
    # this node plots before and after images of masking T1 template
    plot_before_after_mask_t1 = Node(
        SimpleBeforeAfter(), name="plot_before_after_mask_t1"
    )
    # set labels for the before and after images
    plot_before_after_mask_t1.inputs.before_label = "Subject T1"
    plot_before_after_mask_t1.inputs.after_label = "Masked Subject T1"
    # this node plots the masked subject T1 as before and the dwi registered
    # to it as after
    plot_before_after_t1_dwi = Node(
        SimpleBeforeAfter(), name="plot_before_after_t1_dwi"
    )
    # set labels for the before and after images
    plot_before_after_t1_dwi.inputs.before_label = "Masked Subject T1"
    plot_before_after_t1_dwi.inputs.after_label = "Registered DWI"
    # this node plots the extracted brain mask as outline on the initial dwi
    # image
    plot_bet = Node(SimpleShowMaskRPT(), name="plot_bet")
    # this node plots the transformed mask as an outline on transformed dwi
    # image
    plot_transformed = Node(SimpleShowMaskRPT(), name="plot_transformed")

    def ribbon_on_dwi(dwi_file, ribbon_mask):
        import nibabel as nib
        from niworkflows.viz.utils import (
            compose_view,
            cuts_from_bbox,
            plot_registration,
        )

        dwi_img = nib.load(dwi_file)
        ribbon_img = nib.load(ribbon_mask)
        svg = plot_registration(
            dwi_img,
            "Ribbon mask on DWI",
            estimate_brightness=True,
            cuts=cuts_from_bbox(ribbon_img, cuts=7),
            contour=ribbon_img,
        )
        out_file = compose_view(svg, [], out_file="ribbon_on_dwi.svg")

        return out_file

    RibbonOnDWI = Function(
        input_names=["dwi_file", "ribbon_mask"],
        output_names=["out_file"],
        function=ribbon_on_dwi,
    )
    plot_ribbon_on_dwi = Node(RibbonOnDWI, name="plot_ribbon_on_dwi")

    # Create a Merge node to combine the outputs of plot_bet,
    # plot_before_after_eddy, and plot_transformed
    merge_node = Node(Merge(10), name="merge_node")

    # embed plots in a html template
    CreateHTML = Function(
        input_names=[
            "calling_wf_name",
            "report_wf_name",
            "template_path",
            "output_dir",
            "bids_entities",
            "plots",
        ],
        output_names=["out_file"],
        function=create_html_report,
    )
    create_html = Node(CreateHTML, name="create_html")
    create_html.inputs.calling_wf_name = calling_wf_name
    create_html.inputs.report_wf_name = name
    create_html.inputs.template_path = REPORT_TEMPLATE
    create_html.inputs.output_dir = output_dir
    workflow = Workflow(name=name, base_dir=output_dir)
    workflow.connect(
        [
            # plot the extracted brain mask as outline on the initial dwi image
            (
                inputnode,
                plot_bet,
                [
                    ("bet_mask", "mask_file"),
                    ("dwi_initial", "background_file"),
                ],
            ),
            (
                inputnode,
                plot_before_after_mppca,
                [("initial_mean_bzero", "before")],
            ),
            (
                inputnode,
                plot_before_after_mppca,
                [("mppca_denoised", "after")],
            ),
            (
                inputnode,
                plot_before_after_gibbs,
                [("mppca_denoised", "before")],
            ),
            (
                inputnode,
                plot_before_after_gibbs,
                [("gibbs_unringed_denoised", "after")],
            ),
            # plot the initial dwi as before
            (
                inputnode,
                plot_before_after_eddy,
                [("gibbs_unringed_denoised", "before")],
            ),
            # plot the eddy corrected dwi as after
            (
                inputnode,
                plot_before_after_eddy,
                [("eddy_mean_bzero", "after")],
            ),
            # plot the initial subject T1 as before
            (inputnode, plot_before_after_mask_t1, [("t1_initial", "before")]),
            # plot the masked subject T1 as after
            (inputnode, plot_before_after_mask_t1, [("t1_masked", "after")]),
            # plot the masked subject T1 as before and transformed dwi as
            # after
            (inputnode, plot_before_after_t1_dwi, [("t1_masked", "before")]),
            (
                inputnode,
                plot_before_after_t1_dwi,
                [("registered_mean_bzero", "after")],
            ),
            # plot the transformed mask as an outline on transformed dwi image
            (
                inputnode,
                plot_transformed,
                [
                    ("dwi_rigid_registered", "background_file"),
                    ("mask", "mask_file"),
                ],
            ),
            (
                inputnode,
                plot_ribbon_on_dwi,
                [
                    ("registered_mean_bzero", "dwi_file"),
                    ("ribbon_mask", "ribbon_mask"),
                ],
            ),
            # merge the outputs of plot_bet, plot_before_after_eddy,
            # plot_before_after_mask_t1, plot_transformed
            (plot_bet, merge_node, [("out_report", "in1")]),
            (plot_before_after_eddy, merge_node, [("out_report", "in2")]),
            (plot_before_after_mask_t1, merge_node, [("out_report", "in3")]),
            (plot_before_after_t1_dwi, merge_node, [("out_report", "in4")]),
            (plot_transformed, merge_node, [("out_report", "in5")]),
            (
                inputnode,
                merge_node,
                [("plot_recon_surface_on_t1", "in6")],
            ),
            (
                inputnode,
                merge_node,
                [("plot_recon_segmentations_on_t1", "in7")],
            ),
            (plot_ribbon_on_dwi, merge_node, [("out_file", "in8")]),
            (plot_before_after_mppca, merge_node, [("out_report", "in9")]),
            (plot_before_after_gibbs, merge_node, [("out_report", "in10")]),
            # input the bids_entities
            (inputnode, create_html, [("bids_entities", "bids_entities")]),
            # create the html report
            (merge_node, create_html, [("out", "plots")]),
            # output the html report
            (create_html, outputnode, [("out_file", "out_file")]),
        ]
    )
    return workflow

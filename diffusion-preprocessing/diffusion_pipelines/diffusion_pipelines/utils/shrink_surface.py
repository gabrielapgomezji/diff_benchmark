#!/usr/bin/env python
import argparse
import tempfile

import os

import nibabel
import nimesh
import numpy

from diffusion_pipelines.utils import spatial


def shrink_surface(surface_file, reference, distance, outfile):
    """Shrinks the surfaces some mm in the wmmask followind the sig_dis"""

    surface = nimesh.io.load(surface_file)

    # Load white matter mask and its metadata
    wm_mask = nibabel.load(reference)
    voxel_size = wm_mask.header.get_zooms()[:3]

    # Compute a signed distance map inside of the white matter
    tmp_file = tempfile.NamedTemporaryFile(suffix=".nii.gz")

    command_string = "wb_command -create-signed-distance-volume {} {} {}"
    os.system(command_string.format(surface_file, reference, tmp_file.name))

    signed_distance_si = nibabel.load(tmp_file.name)
    signed_distance = signed_distance_si.get_fdata()

    # Transform the points to voxels
    mm_to_voxel_affine = numpy.linalg.inv(wm_mask.affine)
    vertices_voxel = nibabel.affines.apply_affine(
        mm_to_voxel_affine, surface.vertices
    )

    # Push vertices in the white matter following the gradient
    gradient = numpy.array(numpy.gradient(-signed_distance))

    shrinked_vertices_voxel = [
        spatial.grad_descend(p, gradient, distance, voxel_size)
        for p in vertices_voxel
    ]

    # Bring back to mm
    shrinked_vertices = nibabel.affines.apply_affine(
        wm_mask.affine, shrinked_vertices_voxel
    )

    # Save the shrinked surface
    shrinked_surface = nimesh.Mesh(shrinked_vertices, surface.triangles)
    nimesh.io.save(outfile, shrinked_surface)


def command_line_main():
    # Parser
    parser = argparse.ArgumentParser(description="Shrinks a surface")

    parser.add_argument(
        "-surface",
        dest="surface",
        required=True,
        type=str,
        help="surface to shrink",
    )

    parser.add_argument(
        "-reference",
        dest="reference",
        type=str,
        required=True,
        help="reference volume",
    )

    parser.add_argument(
        "-mm",
        dest="distance",
        type=float,
        required=True,
        help="How much to shrink the surface into the white",
    )

    parser.add_argument(
        "-out",
        dest="outfile",
        required=True,
        type=str,
        help="outfile (nifti gii)",
    )

    args = parser.parse_args()

    shrink_surface(args.surface, args.reference, args.distance, args.outfile)

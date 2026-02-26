#!/usr/bin/env python
from setuptools import setup

if __name__ == "__main__":
    setup(
        name="diffusion_pipelines",
        version="0.1",
        description="Diffusion processing pipelines",
        author="Demian Wassermann",
        author_email="demian.wassermann@inria.fr",
        packages=["diffusion_pipelines"],
        entry_points={
            "console_scripts": [
                "shrink_surface=diffusion_pipelines.utils.shrink_surface:command_line_main",
                "diffusion_pipelines=diffusion_pipelines.cli.run:main",
            ]
        },
        install_requires=[
            r.strip() for r in open("requirements.txt").readlines()
        ],
    )

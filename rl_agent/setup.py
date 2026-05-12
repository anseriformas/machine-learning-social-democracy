from setuptools import setup, Extension
import numpy as np

setup(
    name="fast_env",
    version="0.1",
    ext_modules=[
        Extension(
            "fast_env",
            sources=["fast_env.c"],
            include_dirs=[np.get_include()],
        )
    ],
)

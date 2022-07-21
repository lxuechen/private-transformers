import os
import re

import setuptools

# for simplicity we actually store the version in the __version__ attribute in the source
here = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(here, 'private_transformers', '__init__.py')) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

with open(os.path.join(here, 'README.md')) as f:
    readme = f.read()

setuptools.setup(
    name="private_transformers",
    version=version,
    author="Xuechen Li",
    author_email="lxuechen@cs.toronto.edu",
    description="Train Hugging Face transformers with differential privacy.",
    long_description=readme,
    url="https://github.com/lxuechen/private-transformers",
    packages=setuptools.find_packages(exclude=['examples', 'tests']),
    install_requires=[
        "torch>=1.8.0",
        "prv-accountant",
        "transformers>=4.20.1",  # v0.1.0 uses 4.16.2.
        "numpy",
        "scipy",
        "jupyterlab",
        "jupyter",
        "ml-swissknife",
        "opt_einsum",
        "pytest"
    ],
    python_requires='~=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)

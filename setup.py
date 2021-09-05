import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepsnap",
    version="0.2.1",
    author="Jiaxuan You*, Rex Ying*, Xinwei He, Zecheng Zhang",
    author_email="jiaxuan@cs.stanford.edu, rexy@cs.stanford.edu, xhe17@cs.stanford.edu, zecheng@cs.stanford.edu",
    description="Deepsnap package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/snap-stanford/deepsnap",
    packages=setuptools.find_packages(),
    install_requires=[
        'torch',
        'networkx',
        'numpy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    test_suite='nose.collector',
    test_require=['nose'],
    python_requires='>=3.6',
)

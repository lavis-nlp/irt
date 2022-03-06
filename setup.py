from distutils.core import setup

setup(
    name="irt-data",
    version="1.2.1",
    packages=[
        "irt",
        "irt.data",
        "irt.graph",
        "irt.text",
        "irt.common",
    ],
    license="MIT",
    author="Felix Hamann",
    author_email="felix@hamann.xyz",
    description="Inductive Reasoning with Text - Benchmarks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "pyyaml==5.*",
        "networkx==2.*",
        "gitpython==3.*",
        "tqdm",
        "jupyter",
        "tabulate",
        "matplotlib",
    ],
)

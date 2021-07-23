from distutils.core import setup

setup(
    name="irt-data",
    version="1.0",
    packages=["irt"],
    license="MIT",
    author="Felix Hamann",
    author_email="felix@hamann.xyz",
    description="Inductive Reasoning with Text - Benchmarks",
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

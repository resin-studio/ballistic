import setuptools
import os
import pathlib

def all_paths(base_root : str) -> list[str]:
    paths = []
    for root,_,files in os.walk(base_root, topdown=True):
        prefix = (
            ""
            if root == base_root else
	    root[len(base_root) + 1:] + '/'
        )
        for f in files:
            paths.append(prefix + f)
    return paths

def project_path(rel_path : str):
    base_path = os.path.join(pathlib.Path(__file__).parent.absolute(), '..')
    return os.path.abspath(os.path.join(base_path, rel_path))


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ballistic-atx",
    version="0.0.1",
    author="Thomas Logan",
    author_email="thomas.logan.atx@gmail.com",
    description="A stochastic programming and probabilistic specification system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/...",
    project_urls={
        # "Bug Tracker": "https://github.com/...",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=(x := setuptools.find_packages(where="src"), print(x), x)[-1],
    python_requires=">=3.6",
    package_data = {
        'res': all_paths(project_path('res'))
    }
)
import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='multi_cmd',
    version='0.0.1',
    author='Jeffrey Ma',
    author_email='jjma@caltech.edu',
    description='Implementation Code for Multi-Player CMD',
    long_description=long_description,
    long_description_content_type='ext/markdown',
    packages=setuptools.find_packages(),
    install_requires=["torch",
                      "jupyter",
                      "matplotlib",
                      "numpy"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)

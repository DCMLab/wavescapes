import setuptools

# source : https://packaging.python.org/tutorials/packaging-projects/
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wavescapes",
    version="1.0.0",
    author="Cédric Viaccoz",
    author_email="cedric.viaccoz@gmail.com",
    description="Python library to build wavescapes, plots used in musicology.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dcml/wavescapes",
    packages=setuptools.find_packages(),
    install_requires = [
        'numpy',
        'music21',
        'pretty_midi',
        'matplotlib',
        'librosa',
        'scipy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    data_files = [("", ["LICENSE"])],
    python_requires='>=3.6',
)
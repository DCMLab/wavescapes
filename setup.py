import setuptools

# source : https://packaging.python.org/tutorials/packaging-projects/
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wavescape-viaccoz",
    version="0.0.2",
    author="Cédric Viaccoz",
    author_email="cedric.viaccoz@gmail.com",
    description="Python library to build wavescapes, plot used in musicology",
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
    python_requires='>=3.6',
)
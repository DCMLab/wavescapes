# wavescapes


Wavescapes are plots that can visually represent measurements of regularity in music. Those measurements are represented by colors, which are ordered in a hierarchical manner in the plot allowing for all possible subsections of a musical piece to have their measurement beind displayed on the plot. The regularity is measured through the Discrete Fourier Tansform (DFT). Interpretation of the different components outputed by the DFT with respect to the soure musical piece allow high-level analysis of the tonality in the hierarchy of the piece. A publication describing the methodology and many use cases of this visual tool will be published in the future.


### Installation

To install this package the following command has to be issued on a terminal prompt:

```bash
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps wavescape-viaccoz
```

This will install the `wavescape` package from test.pypi, a testing version of pypi, the standard python package distribution service. As such, the dependencies need to be installed separately (consequence of the `--no-deps` flag), and are listed below.


Depending on which default python distribution you have, the `python` command line instead of `python3` would need to be used. Regardless, you need to be aware this library only works with a version of python that is 3.6 or higher.

Below is the list of packages required in order for this library to work. Link to the package's official webpage, and a short description of its usage in this project is specified. Explanations on how to install each of these can be found on each package's hyperlink.

* [numpy](https://numpy.org/) used for vector operations, the DFT operations, and in order to model the wavescapes as a matrix of colored values. 
* [music21](https://web.mit.edu/music21/) used to parse MIDI files and get temporal and pitch informations from them.
* [pretty_midi](https://github.com/craffel/pretty-midi) only used in order to remove percussive tracks for MIDI files.
* [drawSvg](https://pypi.org/project/drawSvg/) the main graphic library. All plots produced from this library are produced using drawSvg. This library provides utilities to draw in SVG (Scalable Vector Graphic) format. SVG was chosen since it only matematically specificies the line and shapes that forms the plot, and then web browsers or specified software take care of doing the rendering. This allows for the plots to have virtually no set resolution and issues that come with rendering/rasterisation.
* [cairosvg (optional)](https://cairosvg.org/) drawSvg uses this library in order to convert the svg images into png. If there is no need to render the plots in png format using this library directly, installation of this package is not required.
* [scipy.io.wavfile](https://kite.com/python/docs/scipy.io.wavfile) used for real audio processing, converts a wav file into an array of raw audio values.
* [librosa](https://librosa.github.io/librosa/) used to produce chromagrams (i.e. pitch class distributions) from real audio.


### documentation
If you read this on the github repo of this project, then you can simply open up `docs/build/index.html` in your favorite browser to access the documentation. 


### usage
If all functions and classes from this package are correctly imported, the short snippet below is an example on how to generate a wavescape plot from a MIDI file and save it to a SVG format. The resulting file can be opened through any modern browser to be viewed.

```python
from wavescape import *

# transforms the MIDI Files into a list of pitch class distribution, each corresponding to a slice of one quarter note from the file.
pc_mat = produce_pitch_class_matrix_from_filename(filepath = 'Bach Prelude in C Major (BWV 846).mid', aw_size = 1.)

# the DFT is applied to each of the pitch class distribution
fourier_mat = apply_dft_to_pitch_class_matrix(pc_mat)

# only the third Fourier coefficient is kept from the previous result and the matrix holding all color coded measurement is built
coeff_mat = complex_utm_to_ws_utm(fourier_mat, coeff=3)

# an instance of a class that allows the drawing of the previous matrix of colors is produced with the resolution being indicated as 500 pixels in width.
ws = Wavescape(coeff_mat, width=500)

# this draw the plot as an SVG image.
canvas = ws.draw()

# saves the produced svg "canvas" into an svg file. If this code is called in a jupyter notebook's cell, just leaving the variable "canvas" at the end of the cell is enough to generate the plot in the cell's output.  
canvas.saveSvg('bach_3rd_coeff_wavescape.svg')
```
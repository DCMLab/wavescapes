# wavescapes


Wavescapes are visual plots which can represent measurement of regularity in music. Those measurements are represented by colors, which are ordered in a hierarchical manner in the plot allowing for all possible subsections of a musical piece to have their measurement of regularity beind displayed on the plot. The regularity is measured through the Discrete Fourier Tansform (DFT). In the case .... (TODO: Complete)



### Installation/dependencies

To install this package the following command has to be issued on a terminal prompt:

```bash
python3 -m pip install --index-url https://test.pypi.org/simple/ wavescape-viaccoz/
```

Depending on which default python distribution you have, the `python` command line instead of `python` would need to be used.

Below is the list of packages required in order for this library to work. Link to the package's official webpage, and a short description of its usage in the project is specified. Normally they get installed at the same time of this package if they are not already present in your environment beforehand.

* [numpy](https://numpy.org/) used for vector operations, the dft operations, and in order to model the wavescapes as matrix of colored values. 
* [music21](https://web.mit.edu/music21/) used to parse MIDI files and get temporal and pitch informations from them.
* [pretty_midi](https://github.com/craffel/pretty-midi) only used in order to remove percussive tracks for MIDI files.
* [drawSvg](https://pypi.org/project/drawSvg/) the main graphic library. All plots produced from this library are produced using drawSvg. This library provides utilities to draw in SVG (Scalable Vector Graphic) format. SVG was chosen since it only matematically specificies the line and shapes that forms the plot, and then web browsers or specified software take care of doing the rendering. This allows for the plots to have virtually no set resolution and issues that come with rendering/rasterisation.
* [cairosvg (optional)](https://cairosvg.org/) drawSvg uses this library in order to convert the svg images into png. If there is no need to render the plots in png format, installation of this package is not required.
* [scipy.io.wavfile](https://kite.com/python/docs/scipy.io.wavfile) used for real audio processing, converts a wav file into an array of raw values.
* [librosa](https://librosa.github.io/librosa/) used to produce chromagrams (i.e. pitch class distributions) from real audio.


### documentation
If you read this on the github repo of this project, then you can simply open up `docs/build/index.html` in your favoite browser to access the documentation. 


### usage
If all functions and classes from the "module" are correctly imported, the short snippet below is an example on how to generate a wavescape plot from a MIDI file:

```python
from wavescape import *
# transforms the MIDI Files into a list of pitch class distribution, each corresponding to a slice of one quarter note from the file.
pc_mat = produce_pitch_class_matrix_from_filename('Bach Prelude in C Major (BWV 846).mid', aw_size = 1.)
# the DFT is applied to each of the pitch class distribution
fourier_mat = apply_dft_to_pitch_class_matrix(pc_mat)
# only the third Fourier coefficient is kept from the previous result and the matrix holding all color coded measurement is built
coeff_mat = complex_utm_to_ws_utm(fourier_mat, 3)
# an instance of a class that allows the drawing of the preivous matrix of colors is produced with the resolution being indicated as 500 pixels in width
ws = Wavescape(coeff_mat, 500)
# this draw the plot as an SVG image.
canvas = ws.draw()
# saves to produced svg "canvas" into an svg file. If this code snipped is called in a jupyter notebook's cell, just leaving the variable "canvas" at the end of the cell is enought to generate the plot in the cell's output.  
canvas.saveSvg('bach_3rd_coeff_wavescape.svg')
```

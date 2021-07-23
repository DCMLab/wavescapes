# wavescapes

![Image showing all six wavescapes produced from Chopin's Prelude in A Minor](img/chopin_prelude_all_coeffs.png?raw=true "Chopin's Prelude in A Minor, visualized by wavescapes")


Wavescapes are plots that can visually represent measurements of regularity in music. Those measurements are represented by colors, which are ordered in a hierarchical manner allowing all possible subsections of a musical piece to have their measurement being displayed on the plot. The regularity is measured through the Discrete Fourier Tansform (DFT). Interpretation of the different components outputed by the DFT with respect to the soure musical piece allow high-level analysis of the tonality in the hierarchy of the piece. [Here is the paper published by Musicae Scientiae describing the methodology and some case studies of this novel method of visualisation](https://www.doi.org/10.1177/10298649211034906).


### Installation

To install this package the following command has to be issued on a terminal prompt:

```bash
pip install wavescapes
```

Depending on which default python distribution you have, `pip3` instead of `pip` might be used. This library only works with a version of python that is 3.6 or higher.

Below is the list of packages required in order for this library to work. Link to the package's official webpage, and a short description of its usage in this project is specified. Explanations on how to install each of these can be found on each package's hyperlink.

* [numpy](https://numpy.org/) used for vector operations, the DFT operations, and in order to model the wavescapes as a matrix of colored values. 
* [music21](https://web.mit.edu/music21/) used to parse MIDI files and get temporal and pitch informations from them.
* [pretty_midi](https://github.com/craffel/pretty-midi) only used in order to remove percussive tracks for MIDI files.
* [matplotlib](https://matplotlib.org/) the main graphic library. 
* [madmom](https://github.com/CPJKU/madmom) used to produce chromagrams (i.e. pitch class distributions) from real audio.
	* (Optional) [ffmpeg](https://ffmpeg.org/) madmom supports only audio files in WAV formats, to enable support for most common audio formats such as `mp3`, `ogg` or `aif`, you need `ffmpeg` which can be used internally by madmom.


### Documentation & How to use

The library is small and lightweight as there are only three functions useful for plotting and understanding wavescapes. However, we do provide a lot of customizations through the means of optional parameters for each functions. This is why we have designed a thorough tutorial on how to use the library and manipulate all the options available. The tutorial consists of a single jupyter notebook located in the `notebook` folder. You can read the tutorial on your browser by following the link below, but we strongly recommand you to clone this repository and to run the tutorials notebook on your own. You can then try to change the parameters and experiment with the different customizations options proposed, or use the notebooks as a base for generating your own wavescapes.

[Link to a notebook viewer of the tutorial](https://nbviewer.jupyter.org/github/DCMLab/wavescapes/blob/production/notebooks/Tutorial.ipynb)

No official documentation is provided, however most of the functions from this package have been described by python docstring you may find in the source code (`wavescapes` folder).

### Quickstart
If all functions and classes from this package are correctly imported, the short function call below is an example on how to generate a wavescape plot from a MIDI file and then save it as a PNG file.

```python
from wavescapes import single_wavescape


single_wavescape(filepath = 'Bach Prelude in C Major (BWV 846).mid',
				 #plot's width in pixels
				 individual_width = 500,
				  #length of the shortest segments of the wavescapes in terms of quarter note
				 aw_size = 1.,
				 #the coefficient to be displayed
				 coefficient=3,
				 #amount of segments per tick drawn on the horizontal axis
				 aw_per_tick=4,
				 #no offset introduced for the position of the ticks, the value 0 also indicates the tick numbers have to start at 1 and not 0.
				 tick_offset=0,
				 #saves the figure drawn as a PNG image.
				 save_label='bach_3rd_coeff_wavescape.png'
				 )
```

![Image showing the result of the code snippet above](img/bach_3rd_coeff_wavescape.png?raw=true)
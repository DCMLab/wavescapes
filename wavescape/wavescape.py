#library dependencies
import numpy as np
import music21 as m21
import pretty_midi as pm
import drawSvg as draw
#drawSvg needs cairosvg in order to output png file. Can work without, if only svg files are used.
from librosa.feature import chroma_stft
from scipy.io import wavfile

#all part of the python standard library
import tempfile
import math
import os

test_midi_folder = 'midiFiles/'
test_audio_folder = 'audioFiles/'

twelve_tones_vector_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#','A', 'A#', 'B']

#this is to correct the name found in the outputed notes from music21 parsiing 
#of MIDI files.
altered_notation_dict = {
    'B#' : 'C',
    'D-' : 'C#',
    'E-' : 'D#',
    'F-' : 'E',
    'E#' : 'F',
    'G-' : 'F#',
    'A-' : 'G#',
    'B-' : 'A#',
    'C-' : 'B'
} 

pitch_pitch_dict = {x: x for x in twelve_tones_vector_name}

#In the end we want each string to match an index between 0 and 11, so it fits inside a 12-d vector.
pitch_index_dict = {twelve_tones_vector_name[i]:i for i in range(len(twelve_tones_vector_name))}

#So any pitch name given to this dict will be mapped to its cannonical form defined in 'twelve_tones_vector_name'
normalize_notation_dict = dict(altered_notation_dict,  **pitch_pitch_dict)

def recursively_map_offset(filepath, only_note_name=True):
    '''
    This function recursively walks through the file's  stream's elements, 
    and whenever it encounters a note, it will append its rhytmic 
    data to the pitch and then store the resulting data structure in an array.
    If a music21 element of type chord is encountered, the chord is decomposed into all
    notes it is composed by, and each of thoses notes are appended the 
    Returns the aforementionned array.
    The rhytmic data is expressed as a tuple of the offset of the beginning of the note
    and the offset of the end of the note.
    
    All temporal informations from MIDI events parsed by the music21 library are encoded
    as unit of quarter notes count, regardless of the BPM+ or the time signature.
    
    Parameters
    ----------
    filepath : str
        the path of the MIDI file to be parsed into the list of pitches and time offset.
        
    only_note_name: bool, optional 
        indicates whether the notes need to be converted from music21 object with octave indication
        to only a string indicating the pitch class name. 
        Default value is True.
        
    Returns
    -------
    iterable of tuples
        the first element of each tuple from this iterable can either be a string indicating the 
        pitch class name of an occurence of one note, or an instance of music21.note representing 
        the occurence of one note, if the parameters 'only_note_name' is set to 'False'. 
        The second half of the tuple, is a tuple itself of float values, the first indicating
        the start offset in time of the note's occurrence, while the second indicates the end
        offset in time of the note's occurrence. Both values are expressed in terms of quarter 
        note length.
        '('E', (10,11,5))' for example would be one element of this iterable, representing an E note
        which starts at the 10th quarter note of the MIDI file and stops one quarter note and half
        later in the piece. 
        
    
    '''
    midi_stream = m21.converter.parse(filepath)
    res = []
    for elem in midi_stream.recurse():
        if isinstance(elem, m21.note.Note):
            start_offset = elem.getOffsetInHierarchy(midi_stream)
            res.append((elem.name if only_note_name else elem, (start_offset, start_offset+elem.duration.quarterLength)))
        elif isinstance(elem, m21.chord.Chord):
            start_offset = elem.getOffsetInHierarchy(midi_stream)
            res += list(map(lambda r: (r.name if only_note_name else r , (start_offset, start_offset+elem.duration.quarterLength)), elem.pitches))
    return res


def remove_drums_from_midi_file(midi_filepath):
    '''
    Takes care of removing drum tracks from a midi filename.
    Work only if the MIDI file has metadata clearly indicating channels that are
    percussive. Does not remove channels of percussive instruments that are pitched
    (like the glockenspiel or the xylophone for instance).  
    
    Parameters
    ----------
    midi_filepath : str  
        the file path of the MIDI file that needs to have percussive channels removed. 
    
    Returns
    -------
    str
        A file path of the same midi file generated without the percussive channels.
        This file path leads to a temporary folder generated on the user's OS file system
        The existence of such temporary folder can not be guaranteed to last for a long
        span of time, this depends on the user's OS bahvior fir temporary folder. As such,
        the resulting file should be used as soone as possible, or moved to a permanet folder.
    
    '''
    sound = pm.PrettyMIDI(midi_filepath)
    
    #getting the track indices of unpitched "percussive" tracks. 
    drum_instruments_index = [idx for idx, instrument in enumerate(sound.instruments) if instrument.is_drum]
    for i in sorted(drum_instruments_index, reverse=True):
        del sound.instruments[i]

    folder = tempfile.TemporaryDirectory()
    temp_midi_filepath = folder.name+'tmp.mid'
    sound.write(temp_midi_filepath)
    
    return temp_midi_filepath

def slice_according_to_beat(pitch_offset_list, beat1_offset, beat2_offset):
    '''
    the beat offset must be expressed as units of quarter notes. 
    Taken are all beat which at least END AFTER the beat1, and START BEFORE the beat2
    '''
    def only_keep_pitches_in_boundaries(pitch_offset_list, beat1_offset, beat2_offset): 
        return list(filter(lambda n: n[1][1] >= beat1_offset and n[1][0] <= beat2_offset, pitch_offset_list))

    res = []
    if beat1_offset >= beat2_offset:
        return res
    for n in only_keep_pitches_in_boundaries(pitch_offset_list, beat1_offset, beat2_offset):
        start_b = n[1][0]
        end_b = n[1][1]
        
        res_n = None
        if start_b >= beat1_offset:
            if end_b > beat2_offset:
                res_n = (n[0], (start_b, beat2_offset))
            else:
                res_n = (n[0], (start_b, end_b))
        elif end_b <= beat2_offset:
            #if start_b < beat1_offset: #of course we are in this case since the first if was not triggered.
            res_n = (n[0], (beat1_offset, end_b))
        else:
            #we are thus in the case the start and end time of the note overshoot the boundaries.
            res_n = (n[0], (beat1_offset, beat2_offset))
        #normally inconsistent results should not happen, but it is possible to have a note with duration equals to 0. This line below prevents that and thus keep the things concise.
        if res_n[1][0] < res_n[1][1]:
            res.append(res_n)
    return res

def sum_into_pitch_class_vector(pitch_offset_list, start_beat, end_beat):
    pitch_class_offset = lambda t: pitch_index_dict[normalize_notation_dict[t[0]]]
    pitch_class_vec = np.zeros(12)
    for tup in pitch_offset_list:
        #we need to be sure we don't take into account the part of the note that exceed the window's size.
        min_beat = max(start_beat, tup[1][0])
        max_beat = min(end_beat, tup[1][1])
        pitch_weight = max_beat - min_beat
        pitch_class_vec[pitch_class_offset(tup)] += pitch_weight
    return pitch_class_vec


def get_max_beat(pitch_offset_list):
    return math.ceil(max(list(map(lambda r: r[1][1], pitch_offset_list))))

def pitch_class_set_vector_from_pitch_offset_list(pitch_offset_array, aw_size=0.5): #the analysis window size (aw_size) is expressed in terms of number of beat (quarter of measures in binary time signature).
    '''
    This functions transforms a list of tuples each containing the name of the pitch
    followed by its start and end in the file into a pitch class distribution with each
    pitch class given the weight corresponding to its duration in the current slice of
    temporal size aw_size.
    
    '''
    
    max_beat = get_max_beat(pitch_offset_array)
    
    if aw_size <= max_beat/2:
        chunk_number = math.ceil(max_beat/aw_size)
    else:
        raise Exception('The analysis window\'s size should not exceed half the duration of the musical piece.')
    
    res_vector = np.full((chunk_number, 12), 0.0, np.float64)

    for b in range(chunk_number):
        start_beat = b*aw_size
        stop_beat = (b+1)*aw_size
        analysis_windows = slice_according_to_beat(pitch_offset_array, start_beat, stop_beat)
        pitch_class_vec = sum_into_pitch_class_vector(analysis_windows, start_beat, stop_beat)
        res_vector[b] = pitch_class_vec
    
    return res_vector


# trim the input array so that no empty vectors are located at the beginning and end of the muscial piece
def trim_pcs_array(pcvs):
    start = 0
    while not np.any(pcvs[start]):
        start += 1
    end = len(pcvs) - 1
    while not np.any(pcvs[end]):
        end -= 1
    return pcvs[start:end+1]


def audio_int_to_float32(data, byte_width):
    return data.astype(np.float32, order='C') / float(2**(byte_width-1))
    
def audiowav_to_mono_float_array(filename):
    sr, data = wavfile.read(filename)
    data_dtype_str = str(data.dtype)
    if data_dtype_str[:3] == 'int':
        #converting the array into float
        data = audio_int_to_float32(data, int(data_dtype_str[3:]))
    elif data_dtype_str[:5] != 'float':
        raise Exception('wav audio data is not in a format that can (trivially) be converted into float32')
    channel_nbr = np.shape(data)[1]
    if channel_nbr > 1:
        #(super)stereo signal need to be converted into mono.
        data = data.sum(axis=1) / channel_nbr
    return data, sr

# "aw" in "number_of_aw" stands for analysis window. It determines how many chromagrams the signal will be discreticized into. 
def produce_chromagrams_from_audio_file(audio_filename, aw_size):
    audio_array, sample_ratio = audiowav_to_mono_float_array(audio_filename)
    
    hop_len = sample_ratio*aw_size #hop_len is the analysis window size for the chromagrams in terms of number of sample.
    #so the result's shape is consistent with the one produced in the case of midi files.
    return np.transpose(chroma_stft(audio_array, sample_ratio, hop_length=hop_len))

def produce_pitch_class_matrix_from_filename(filepath, remove_percussions = True, aw_size = 1., trim_extremities=True):
    '''
    This function takes a MIDI or WAV file as a parameters and
    transforms it into "list of pitch class distribution"
    This list is modelised by a Nx12 matrix of float values, each 
    row of the matrix corresponds to the pitch content of one slice
    of temporal size "aw_size" from the musical content from the
    input's file. The number of row N depends on the temporal size
    of the file, and the number chosen for aw_size.
    
    Parameters
    ----------
    filepath : str 
        the path of the MIDI/WAV file whose musical content 
        is transformed into the pitch class distribution's list
                 
    remove_percussions : bool, optional 
        indicates whether percussive instruments need to be removed from the mix. 
        ONLY APPLIES TO MIDI FILES CURRENTLY.
        Default value is True.
                          
    aw_size : float, optional
        means "analysis window's size", represent the size of each
        slice of pitch content from the file in terms of time. In the case 
        of MIDI file, this number represent the number of beat (a beat being
        a quarter note), and in the case of WAV file, this number represents 
        the number of seconds.
        Default value is 1 (1 quarter note, or 1 second depending on the file's type)
                
    trim_extremities : bool, optional
        inidicates whether the silences that are present at both
        extremities needs to be removed from the resulting list of pitch
        class distribution. ONLY APPLIES TO MIDI FILES CURRENTLY
        Default value is True.
        
    
    Returns
    -------
    numpy matrix of shape Nx12 (numpy.ndarray of numpy.float64)
        This matrix holds the pitch distributions corresponding to all
        the pitch content from all non overlapping slices of aw_size size from the file
        given as argument.
    
    '''
    if filepath.endswith('.mid') or filepath.endswith('.midi'):
        midi_filepath = remove_drums_from_midi_file(filepath) if remove_percussions else filepath
        pitch_offset_list = recursively_map_offset(midi_filepath)
        pcvs_arr = pitch_class_set_vector_from_pitch_offset_list(pitch_offset_list, aw_size)
        return trim_pcs_array(pcvs_arr) if trim_extremities else pcvs_arr
    elif filename.endswith('.wav'):
        return produce_chromagrams_from_audio_file(filename, aw_size)
    else:
        raise Exception('The file should be in MIDI or WAV format')
        
    return recursively_map_offset(midi_stream)
    
    
def build_utm_from_one_row(res):
    '''
    given a NxN matrix whose first row is the only
    one that's filled with values, this function fills
    all the above row by summing for each row's element
    the two closest element from the row below. This
    method of summing builds an upper-triangle-matrix
    whose structure represent all hierarchical level.
    '''
    pcv_nmb = np.shape(res)[0]
    for i in range(1, pcv_nmb):
        for j in range(0, pcv_nmb-i):
            res[i][i+j] = res[0][i+j] + res[i-1][i+j-1]
    return res

def apply_dft_to_pitch_class_matrix(pc_mat, build_utm = True):
    '''
    This functions takes a list of N pitch class distributions,
    modelised by a matrix of float numbers, and apply the 
    DFT individually to all the pitch class distributions.
    
    Parameters
    ----------
    pc_mat: numpy matrix of shape (N, 12) (numpy.ndarray of numpy.float64)
        holds the pitch class distribution of all slice of a minimum temporal size.
    build_utm: bool, optional 
        indicates whether the resulting list of DFT results need to be built into an upper 
        triangle matrix representing all hierarchical levels possible from the original musical piece.
        Default value is True.
    
    Returns
    -------
    numpy matrix (numpy.ndarray of numpy.complex128)
        according to the parameters 'build_utm', either a Nx7 complex number matrix being
        the converted input matrix of pitch class distribution
        transformed into Fourier coefficient, or a NxNx7 complex number 
        upper triangle matrix being the fourier coefficient obtained from all
        possible slices of the original musical piece.
    '''
    pcv_nmb, pc_nmb = np.shape(pc_mat)
    #+1 to hold room for the 0th coefficient
    coeff_nmb = int(pc_nmb/2)+1
    res_dimensions = (pcv_nmb, coeff_nmb)
    res = np.full(res_dimensions, (0. + 0.j), np.complex128)

    for i in range(pcv_nmb):
        res[i] = np.fft.fft(pc_mat[i])[:coeff_nmb] #coeff 7 to 11 are uninteresting (conjugates of coeff 6 to 1).
    
    if build_utm:
        new_res = np.full((pcv_nmb, pcv_nmb, coeff_nmb), (0. + 0.j), np.complex128)
        new_res[0] = res 
        res = build_utm_from_one_row(new_res)
        
    return res

def rgba_to_rgb(to_convert, background):
    if len(to_convert) == 3:
        return to_convert #no point converting something that is already in RGB
    if len(to_convert) != 4:
        raise Exception('Incorrect format for the value to be converted, should have length of 4')
    if len(background) != 3:
        raise Exception('Incorrect format for the value background, should have length of 3 (no alpha channel for this one)')
    alpha = float(to_convert[3])/255.0
    return [int((1 - alpha) * background[i] + alpha * to_convert[i]) for i in range(len(background))]

stand = lambda v: int(v*0xff)

def circular_hue(angle, opacity=0xff, output_rgb=True, needs_shifting=True):
    def step_function_quarter_pi_activation(lo_bound, hi_bound, value):
        #in the increasing path branch
        if value >= lo_bound and value <= lo_bound + math.pi/3:
            return ((value-lo_bound)/(math.pi/3))
        #in the decreasing path branch
        elif value >= hi_bound and value <= hi_bound + math.pi/3:
            return 1-((value-hi_bound)/(math.pi/3))
        else:
            #the case of red 
            if lo_bound > hi_bound:
                return 0 if value > hi_bound and value < lo_bound else 1
            else:
                return 1 if value > lo_bound and value < hi_bound else 0
    #Need to shift the value with one pi as the range of the angle given is between pi and minus pi
    #and the formulat I use goes from 0 to 2pi.
    angle = angle + math.pi if needs_shifting else angle 
    green = lambda a: step_function_quarter_pi_activation(0, math.pi, a)
    blue = lambda a: step_function_quarter_pi_activation(math.pi*2/3, math.pi*5/3, a)
    red = lambda a: step_function_quarter_pi_activation(math.pi*4/3, math.pi/3, a)
    value = (stand(red(angle)), stand(green(angle)), stand(blue(angle)), opacity)
    if output_rgb:
        value = rgba_to_rgb(value, background=(0xff,0xff,0xff))
    return value

def complex_utm_to_ws_utm(utm, coeff, magn_stra = '0c', segmented_mode = False):
    '''
    This function converts an upper triangle matrix filled with Fourier coefficient into 
    an upper triangle matrix filled with color values that serves as the mathematical model
    holding the color information needed to build the plot called wavescape.
    
    Parameters
    ----------
    utm : numpy matrix shape NxN (numpy.ndarray of numpy.complex128)
        An upper triangle matrix holding all fourier coefficients 0 to 6 at all different hierarchical levels
    
    coeff: int
        number between 1 to 6, will define which coefficient plot will be visualised in the outputted upper triangle matrix. 
    
    magn_strat : {'0c', 'max', 'max_weighted'}, optional
        Since the magnitude is unbounded, but its grayscale visual representation needs to be bounded,
        Different normalisation/clamping of the magnitude can be possible to constrain it to a value between 0 and 1.
        Below is the listing of the different value accepted for the argument magn_stra
        - '0c' : default normalisation, will normalise each magnitude by the 0th coefficient 
            (which corresponds to the sum of the weight of each pitch class). This ensures only
            pitch class distribution which exactly match the periodicity of the coefficient can
            reach the value of 1.
        - 'max': set the grayscal value 1 to the maximum possible magnitude in the wavescape, and interpolate linearly
            all other values of magnitude based on that max set to 1. Warning: will bias the visual representation
            in a way that the top of the visualisation will display much more magnitude than lower levels. 
        - 'max_weighted': same principle as max, except the maximum magnitude is now taken at the hierarchical level,
            in other words, the magnitude is relative to the max
        Default value is '0c'
                      
    segmented_mode : bool, optional 
        Determines wether the outputted matrix contains RGB (False) or RGBA (True). In segmented 
        mode, the alpha channel holds the magnitude information, while the RGB holds the phase
        In segmented mode the magnitude is applied to the RGB color of the phase to produce a
        single RGB value that corresponds to the blend between the hue mapping of the phase
        and the grayscale mapping of the magnitude.
        Default value is False.
    
    Returns
    -------
    numpy.ndarray
        an upper triangle matrix of dimension NxN holding all rgb (or rgba depending on the value of segmented mode) 
        values corresponding to the color mapping of a single Fourier coefficient from the input.
    '''
    
    
    
    def zeroth_coeff_cm(value, coeff, ):
        zero_c = value[0].real
        if zero_c == 0.:
            #empty pitch class vector, thus returns white color value.
            #this avoid a nasty divide by 0 error two lines later.
            return (0.,0.)#([0xff]*3
        nth_c = value[coeff]
        magn = np.abs(nth_c)/zero_c
        angle = np.angle(nth_c)
        return (angle, magn)#circular_hue(angle, opacity=stand(magn))
    
    def max_cm(value, coeff, max_magn):
        if max_magn == 0.:
            return (0.,0.)
        nth_c = value[coeff]
        magn = np.abs(nth_c)
        angle = np.angle(nth_c)
        return (angle, magn/max_magn)
    
    
    shape_x, shape_y = np.shape(utm)[:2]
    channel_nbr = 4 if segmented_mode else 3
    res = np.full((shape_x, shape_y, channel_nbr), (0xff), np.uint8)
    
    if magn_stra == '0c':
        for y in range(shape_y):
            for x in range(shape_x):
                angle, magn = zeroth_coeff_cm(utm[y][x], coeff)
                res[y][x] = circular_hue(angle, opacity=stand(magn), output_rgb = not segmented_mode)
                
    elif magn_stra == 'max':
        #arr[:,:,coeff] is a way to select only one coefficient from the tensor of all 6 coefficients 
        max_magn = np.max(np.abs(utm[:,:,coeff]))
        for y in range(shape_y):
            for x in range(shape_x):
                angle, magn = max_cm(utm[y][x], coeff, max_magn)
                res[y][x] = circular_hue(angle, opacity=stand(magn), output_rgb = not segmented_mode)
                
    elif magn_stra == 'max_weighted':
        for y in range(shape_y):
            line = utm[y]
            max_magn = np.max([np.abs(el[coeff]) for el in line])
            for x in range(shape_x):
                res[y][x] = max_cm(utm[y][x], coeff, max_magn)
                res[y][x] = circular_hue(angle, opacity=stand(magn), output_rgb = not segmented_mode)
    else:
        print('Unknown option for magn_stra')
    
    return res

SQRT_OF_THREE = math.sqrt(3)


def rgb_to_hex(rgb):
    if type(rgb) is str and rgb[0] == '#' and len(rgb) > 6:
        # we already have an hex value given let's just return it back.
        return rgb 
    elif len(rgb) == 3:
        return '#%02x%02x%02x' % (rgb[0],rgb[1],rgb[2])
    elif len(rgb) == 4:
        return '#%02x%02x%02x%02x' % (rgb[0],rgb[1],rgb[2], rgb[3])
    else:
        raise Exception('Cannot convert RGB tuple to hex value if the value given is neither in the RGB or the RGBA format.')

class Diamond(object):
    def __init__(self, x, y, width, height, color):
        self.half_width = width/2.
        self.half_height = height/2.
        self.x = x
        self.y = y
        self.color = color
        
    def draw(self, canvas, new_color=None, stroke=None):
        new_color = new_color if new_color else self.color
        stroke = stroke if stroke else self.color
        d_x = self.x
        d_y = self.y
        canvas.append(draw.Lines(d_x-self.half_width, d_y,
                    d_x, d_y+self.half_height,
                    d_x+self.half_width, d_y,
                    d_x, d_y-self.half_height,
            fill=new_color,
            stroke=stroke))
        
        
class Hexagon(object):
    def __init__(self, x, y, width, color):
        self.w = width/2.
        self.h = SQRT_OF_THREE*self.w/3.
        self.x = x
        self.y = y
        self.color = color
        
    def draw(self, canvas, new_color=None, stroke=None):
        new_color = new_color if new_color else self.color
        stroke = stroke if stroke else self.color
        d_x = self.x
        d_y = self.y
        w = self.w
        h = self.h
        canvas.append(draw.Lines(d_x+w, d_y+h,
                    d_x, d_y+2*h,
                    d_x-w, d_y+h,
                    d_x-w, d_y-h,
                    d_x, d_y-2*h,
                    d_x+w, d_y-h,
            fill=new_color,
            stroke=None))

class Wavescape(object):
    '''
    This class represent an object that holds the attributes 
    and methods needed to effectively draw the plot name "wavescape"

    Attributes
    ----------
    utm : NxNx3 or NxNx4 matrix (numpy.ndarray of numpy.uint8), 
        upper triangle matrix holding color values as tuples of 3 (RGB) or 4 (RGBA) 8 bit integers. 
        Holds the color information and their relevant informations to draw the plot.
        
    width : int
        the width in pixels of the plot. It needs to be at least twice as big as the shape of the 
        upper triangle matrix. The height of the plot is defined by the drawing primitive chosen.
        
    drawing_primitive : {'diamond', 'rhombus', 'hexagon'} , optional 
        the drawing shape that forms a single colored element from the plot. Three primitives are 
        currently available:
          -'diamond': diamond whose height is twice its width
          -'rhombus': diamond formed by two equilateral triangles. Each side is the same size
          -'hexagon': a hexagon, i.e. a 6 sides polygonal shape, each side being the same size.
        default value is 'rhombus'
    '''
    
    
    #Formula derived with a bit of algebra in order to determine the height of the wavescape hexagon plot 
    #based on the just the given plot's width (wi) and the number of layer (n). The SQRT_OF_THREE*wi was broadcasted
    #to the two parts of the addition to mitigate the numeric error caused by the division by 6 times the number
    #of layer (n). Side Note: the self argument is useless there (present just so that the lambda can be called as a method)
    HEXAGON_PLOT_HEIGHT = lambda wi, n: (SQRT_OF_THREE*wi)*(0.5) + ((SQRT_OF_THREE/6.)*(wi/n))
    
    #constants 
    # fun fact that would please anyone with OCD: all drawing primitive names have the same amount of letters.
    DIAMOND_STR = 'diamond'
    RHOMBUS_STR = 'rhombus'
    HEXAGON_STR = 'hexagon'
    
    
    def __init__(self, utm, width, drawing_primitive='rhombus'):
        self.utm = utm
        self.width = width
        self.drawing_primitive = drawing_primitive
        
        mat_dim, mat_dim_other_axis, mat_depth = utm.shape
        if mat_dim != mat_dim_other_axis:
            raise Exception("The upper triangle matrix is not a square matrix")
        if mat_dim > width/2:
            raise Exception("The number of elements to be drawn exceeds the keyscape's resolution.(%d elements out of %d allowed by the resolution) Increase the width of the plot to solve this issue" % (mat_dim, self.width/2))
        if (mat_depth < 3 or mat_depth > 4):
            raise Exception("The upper triangle matrix given as argument does not hold either RGB or RGBA values")
        self.mat_dim = mat_dim
        
        #building a matrix with None to hold the element object for drawing them later.
        self.matrix_primitive = np.full((mat_dim, mat_dim), None, object)
        
        if drawing_primitive == self.RHOMBUS_STR or drawing_primitive == self.DIAMOND_STR:
            self.generate_diamonds()
        elif drawing_primitive == self.HEXAGON_STR:
            self.generate_hexagons()
        else:
            pass
    
    def generate_hexagons(self):
        hexagon_width = self.width/float(self.mat_dim)
        hexagon_height = 2*SQRT_OF_THREE*hexagon_width/3.
        half_width_shift = self.width/2.
        half_height_shift = (Wavescape.HEXAGON_PLOT_HEIGHT(self.width, self.mat_dim))/2.
        
        for y in range(self.mat_dim):
            for x in range(y, self.mat_dim):
                
                curr_color = rgb_to_hex(rgba_to_rgb(self.utm[y][x], background=(0xff,0xff,0xff)))
                #Useless to draw if there is nothing but blank to draw, duh.
                if curr_color != '#FFFFFF':
                    #classic x-axis placement taking into account the half width of the hexagon
                    d_x = hexagon_width/2. + hexagon_width*x
                    #Now shifting all of this to the left to go from utm placement to pyramid placement
                    d_x = d_x - hexagon_width*y/2.
                    #And finally shifting this to take into account drawSvg center placement I posed
                    d_x = d_x - half_width_shift
                    
                    d_y = hexagon_height/2.+(0.75*hexagon_height)*y
                    d_y = d_y - half_height_shift
                    self.matrix_primitive[y][x] = Hexagon(d_x, d_y, hexagon_width, curr_color)
    
    def generate_diamonds(self):
        diamond_width = self.width/float(self.mat_dim)
        diamond_height = diamond_width*2 if self.drawing_primitive != 'rhombus' else diamond_width * SQRT_OF_THREE
        
        half_width_shift = self.width/2.
        half_height_shift = half_width_shift if self.drawing_primitive == 'diamond' else self.width * SQRT_OF_THREE/4.
        
        for y in range(self.mat_dim):
            for x in range(y, self.mat_dim):
                
                curr_color = rgb_to_hex(rgba_to_rgb(self.utm[y][x], background=(0xff,0xff,0xff)))
                #Useless to draw if there is nothing but blank to draw, duh.
                if curr_color != '#FFFFFF':
                    #classic x-axis placement taking into account the edge from the diamond 
                    d_x = diamond_width/2. + diamond_width*x
                    #Now shifting all of this to the left to go from utm placement to pyramid placement
                    d_x = d_x - diamond_width*y/2.
                    #And finally shifting this to take into account drawSvg center placement I posed
                    d_x = d_x - half_width_shift
                    
                    d_y = diamond_height/2.*y
                    d_y = d_y - half_height_shift
                    self.matrix_primitive[y][x] = Diamond(d_x, d_y, \
                                          diamond_width, diamond_height, curr_color)

    def draw(self, plot_indicators = True, add_line = False, add_bg=True):
        '''
        After being called on a properly initialised instance of a Wavescape object,
        this method draws the visual plot known as "wavescape" and return it 
        as a picture in SVG format.

        Parameters
        ----------
        plot_indicators: bool, optional 
            indicates whether rounded indicators on the lateral edges of the plot need to 
            be drawn. A rounded indicator is drawn at each eight of the height of the plot
            Default value is True
            
        add_line: bool, optional
            indicates whether all element of the plot (single drawing primitives) need to be
            outlined with a black line.
            Default value is False.
            
        add_bg: bool, optional
            indicates whether a white background needs to be drawn behind the plot. The produced
            plot being placed on a square
            Default value is True

        Returns
        -------
        drawSvg.drawing.Drawing
            A drawSvg canvas holding the drawn plot. This canvas can be saved in a svg file (using the saveSvg method
            from drawSvg.drawing.Drawing), or a png file (using the savePng method from drawSvg.drawing.Drawing)
            if the package cairosvg is installed and is functionning correctly.
            On jupyter notebook, this canvas can be directly seen as an output of a cell, either by
            using the 
        '''
        if self.matrix_primitive is None or self.matrix_primitive.shape[0] < 1 or self.matrix_primitive.shape[1] < 1:
            raise Exception("cannot draw when there are nothing to draw. Don't forget to generate diamonds in the correct mode before drawing.")

        height = self.width 
        if self.drawing_primitive == 'rhombus':
            height *= SQRT_OF_THREE/2.
        elif self.drawing_primitive == 'hexagon':
            height = Wavescape.HEXAGON_PLOT_HEIGHT(self.width, self.mat_dim)
        width = self.width
        canvas = draw.Drawing(width, height, origin='center')

        if add_bg:
            #manually creating the white background
            canvas.append(draw.Rectangle(-self.width/2, -height/2, self.width, height, fill='white'))

        black = '#000000ff'

        for y in range(self.matrix_primitive.shape[0]):
            for x in range(y, self.matrix_primitive.shape[1]):
                element = self.matrix_primitive[y][x]
                if element:
                    if add_line:
                        element.draw(canvas, stroke=black)
                    else:
                        element.draw(canvas)

        if plot_indicators:
            ind_width = width if self.drawing_primitive != self.HEXAGON_STR else width + 2
            mid_size = int(self.width / 50.)
            eigth_size = int(mid_size /4.)
            quart_size = eigth_size * 3

            white_fill = '#ffffff00'
            middle_gray= '#666666ff'

            params = [
                {'size': mid_size,   'fill': white_fill , 'fill_opacity': 0, 'stroke': 'black' },
                {'size': quart_size, 'fill': white_fill,  'fill_opacity': 0, 'stroke': middle_gray},
                {'size': eigth_size, 'fill': middle_gray, 'fill_opacity': 1, 'stroke': middle_gray}
            ]

            stroke_width = int(self.width / 1000) + 2

            
            # Code to draw the indicators using circles.
            # This is probably the most far fetched discrete mathematical formula I ever made.
            # Basically I found the coordinates relative to the height and width of the plot by trial 
            # and error using negative power of 2, and then I derived a discrete formula
            # depending on two parameters n and m (the second one depending on the first)
            # which give me automatically the right x and y coordinates. It works, just trust me.
            for n in range(1,4):
                p = params[n-1]
                for m in range(2**(n-1)):
                    x = 1/float(2**(n+1)) + m/float(2**n)
                    y = (2**n - 1)/float(2**n) - m/float(2**(n-1)) - 1/2.
                    for i in [-1, 1]:
                        canvas.append(draw.Circle(i*x*width, y*height, p['size'], fill=p['fill'], \
                                                  fill_opacity=p['fill_opacity'], stroke=p['stroke'], \
                                                  stroke_width=stroke_width))

        return canvas
        
    
    def draw_single_stripe(self, stripe_lvl, plot_delimiters=False, square_height_proportion = 1.):
        '''
        draws a single stripe of the wavescape as a line of colored rectangles
        
        Parameters
        ----------
        'stripe_lvl': int 
            indicates which level needs to be drawn as a stripe with the convention that 0
            is the lowest level.
        
        'plot_delimiters': bool, optional 
            indicates whether the rectangles are drawn with black outlines.
            Default value is False
        
        'square_height_proportion': float, optional
            represent how higher the height of the rectangle is with respect
            to his width. The individual width of each rectangle is defined
            by the attribute 'width' of the current instance, and the number 
            of elements to be drawn in the current level
            Default value is 1 (which produces squares as individual elements
            from the stripe).
            
        Returns
        -------
        drawSvg.drawing.Drawing
            A drawSvg canvas holding the drawn plot. This canvas can be saved in a svg file (using the saveSvg method
            from drawSvg.drawing.Drawing), or a png file (using the savePng method from drawSvg.drawing.Drawing)
            if the package cairosvg is installed and is functionning correctly
        '''
        
        if stripe_lvl >= self.mat_dim:
            raise Exception("The stripe's level chosen exceeds the number of layers in the plot (layers 0 to %d accessible)"%self.mat_dim-1)
        square_number = self.mat_dim - stripe_lvl
        square_width = self.width/square_number
        square_height = square_width * square_height_proportion
        #to account for a little bit of room to draw line delimiters on edges of the plot
        margin = 2 if plot_delimiters else 0
        canvas = draw.Drawing(self.width+margin, square_height+margin, origin='center')
        for i in range(0, self.mat_dim - stripe_lvl):
            curr_elem = self.utm[stripe_lvl][stripe_lvl+i]
            curr_color = rgb_to_hex(rgba_to_rgb(curr_elem, background=(0xff,0xff,0xff)))
            hsw = square_width/2.
            hsh = square_height/2.
            x = i*square_width + hsw - self.width/2.
            stroke = '#000000ff' if plot_delimiters else curr_color
            canvas.append(draw.Lines(x-hsw, -hsh,
                    x-hsw, hsh,
                    x+hsw, hsh,
                    x+hsw, -hsh,
                    x-hsw, -hsh, #duplicate of the first point so that it closes the square if lines are drawn
                fill=curr_color,
                stroke=stroke))
        return canvas
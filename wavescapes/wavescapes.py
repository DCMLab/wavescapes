#library dependencies
import numpy as np
import music21 as m21
import pretty_midi as pm
from librosa.feature import chroma_stft
from scipy.io import wavfile
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.ticker import MultipleLocator, IndexLocator,FuncFormatter

#all part of the python standard library
import tempfile
import math
import os
import matplotlib.pyplot as plt


# ---------------------------------------
# Part 1 : Pitch Class Vectors
# ---------------------------------------


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
        extremities of the musical piece needs to be removed from the resulting list of pitch
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



# ---------------------------------------
# Part 2 : DFT and UTM
# ---------------------------------------


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
        As the DFT is linear, the computation of all hierarchical levels can be done at a later sate,
        thus saving some space (O(n) instead of O(n^2)).
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

# implemented following this code : https://alienryderflex.com/saturation.html
def rgb_to_saturated_rbg(rgb_value, saturation_val):
    assert(saturation_val >= 0)
    assert(saturation_val <= 1)
    Pr=.299
    Pg=.587
    Pb=.114
    r,g,b = rgb_value
    P = math.sqrt(((r**2)*Pr)+((g**2)*Pg)+((b**2)*Pb))
    apply_sat = lambda v: P+(v-P)*saturation_val
    
    return (apply_sat(r), apply_sat(g), apply_sat(b))

def circular_hue(angle, magnitude=1., opacity_mapping=True):
    
    #np.angle returns value in the range of [-pi : pi], where the circular hue is defined for 
    #values in range [0 : 2pi]. Rather than shifting by a pi, the solution is for the negative
    #part to be mapped to the [pi: 2pi] range which can be achieved by a modulo operation.
    def two_pi_modulo(value):
        return np.mod(value, 2*math.pi)
    
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
    angle = two_pi_modulo(angle)
    green = lambda a: step_function_quarter_pi_activation(0, math.pi, a)
    blue = lambda a: step_function_quarter_pi_activation(math.pi*2/3, math.pi*5/3, a)
    red = lambda a: step_function_quarter_pi_activation(math.pi*4/3, math.pi/3, a)
    value = None
    if opacity_mapping:
        value = (stand(red(angle)), stand(green(angle)), stand(blue(angle)), stand(magnitude))
        #defautl background for the opacity is white.
        value = rgba_to_rgb(value, background=(0xff,0xff,0xff))
    else:
        value = (stand(red(angle)), stand(green(angle)), stand(blue(angle)))
        value = rgb_to_saturated_rbg(value, magnitude)
    return value


# ---------------------------------------
# Part 3 : Color Mapping
# ---------------------------------------

def complex_utm_to_ws_utm(utm, coeff, magn_stra = '0c', opacity_mapping=True):
    '''
    Converts an upper triangle matrix filled with Fourier coefficients into 
    an upper triangle matrix filled with color values that serves as the mathematical model
    holding the color information needed to build the wavescapes plot.
    
    Parameters
    ----------
    utm : numpy matrix shape NxN (numpy.ndarray of numpy.complex128)
        An upper triangle matrix holding all fourier coefficients 0 to 6 at all different hierarchical levels
    
    coeff: int
        number between 1 to 6, will define which coefficient plot will be visualised in the outputted upper triangle matrix. 
    
    magn_strat : {'0c', 'max', 'max_weighted'}, optional
        Since the magnitude is unbounded, but its grayscale visual representation needs to be bounded,
        Different normalisation of the magnitude are possible to constrain it to a value between 0 and 1.
        Below is the listing of the different value accepted for the argument magn_stra
        - '0c' : default normalisation, will normalise each magnitude by the 0th coefficient 
            (which corresponds to the sum of the weight of each pitch class). This ensures only
            pitch class distribution whose periodicity exactly match the coefficient's periodicity can
            reach the value of 1.
        - 'max': set the grayscal value 1 to the maximum possible magnitude in the wavescape, and interpolate linearly
            all other values of magnitude based on that maximum value set to 1. Warning: will bias the visual representation
            in a way that the top of the visualisation will display much more magnitude than lower levels. 
        - 'max_weighted': same principle as max, except the maximum magnitude is now taken at the hierarchical level,
            in other words, each level will have a different opacity mapping, with the value 1 set to the maximum magnitude
            at this level. This normalisation is an attempt to remove the bias toward higher hierarchical level that is introduced
            by the 'max' magnitude process cited previously.
        Default value is '0c'
                      
    output_opacity : bool, optional 
        Determines whether the normalized magnitude from the fourier coefficients held in the upper-triangle matrix
        "utm" are color-mapped to the opacity of the underlying phase color, or its saturation. 
        Default value is True (i.e. opacity mapping).
    
    Returns
    -------
    numpy.ndarray
        an upper triangle matrix of dimension NxN holding all rgb
        values corresponding to the color mapping of a single Fourier coefficient from the input.
    '''
    
    def zeroth_coeff_cm(value, coeff):
        zero_c = value[0].real
        if zero_c == 0.:
            #empty pitch class vector, thus returns white color value.
            #this avoid a nasty divide by 0 error two lines later.
            return (0.,0.)#([0xff]*3
        nth_c = value[coeff]
        magn = np.abs(nth_c)/zero_c
        angle = np.angle(nth_c)
        return (angle, magn)
    
    def max_cm(value, coeff, max_magn):
        if max_magn == 0.:
            return (0.,0.)
        nth_c = value[coeff]
        magn = np.abs(nth_c)
        angle = np.angle(nth_c)
        return (angle, magn/max_magn)
    
    
    shape_x, shape_y = np.shape(utm)[:2]
    channel_nbr = 3
    res = np.full((shape_x, shape_y, channel_nbr), (0xff), np.uint8)
    
    if magn_stra == '0c':
        for y in range(shape_y):
            for x in range(shape_x):
                angle, magn = zeroth_coeff_cm(utm[y][x], coeff)
                res[y][x] = circular_hue(angle, magnitude=magn, opacity_mapping = opacity_mapping)
                
    elif magn_stra == 'max':
        #arr[:,:,coeff] is a way to select only one coefficient from the tensor of all 6 coefficients 
        max_magn = np.max(np.abs(utm[:,:,coeff]))
        for y in range(shape_y):
            for x in range(shape_x):
                angle, magn = max_cm(utm[y][x], coeff, max_magn)
                res[y][x] = circular_hue(angle, magnitude=magn, opacity_mapping = opacity_mapping)
                
    elif magn_stra == 'max_weighted':
        for y in range(shape_y):
            line = utm[y]
            max_magn = np.max([np.abs(el[coeff]) for el in line])
            for x in range(shape_x):
                angle, magn = max_cm(utm[y][x], coeff, max_magn)
                res[y][x] = circular_hue(angle, magnitude=magn, opacity_mapping = opacity_mapping)
    else:
        raise Exception('Unknown option for magn_stra')
    
    return res

# ---------------------------------------
# Part 3 : Drawing functions
# ---------------------------------------


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

def coeff_nbr_to_label(k):
    if type(k) == str:
        k = int(k)
    if k == 1:
        return '%dst'%k
    elif k == 2:
        return '%dnd'%k
    elif k == 3:
        return '%drd'%k
    else:
        return '%dth'%k
    

def compute_height(width, mat_dim, drawing_primitive):
    if drawing_primitive == Wavescape.HEXAGON_STR:
        return Wavescape.HEXAGON_PLOT_HEIGHT(width, mat_dim)
    elif drawing_primitive ==  Wavescape.RHOMBUS_STR:
        return (width/2.) * SQRT_OF_THREE
    elif drawing_primitive ==  Wavescape.DIAMOND_STR:
        return width
    else:
        raise Exception('Unknown drawing primitive: %s'%drawing_primitive)

class DiamondPrimitive(object):
    def __init__(self, x, y, width, height, color, bottom_diamond):
        self.half_width = width/2.
        self.half_height = height/2.
        self.x = x
        self.y = y
        self.color = color
        self.bottom_diamond = bottom_diamond
        
    def draw(self, new_color=None, stroke=None):
        curr_color = new_color if new_color else self.color
        x = self.x
        y = self.y
        # this is to treat the bottom diamond that needs to be drawn as a triangle
        last_coord = (x,y if self.bottom_diamond else y-self.half_height)
        return Polygon(((x-self.half_width, y),
                               (x, y+self.half_height),
                               (x+self.half_width, y),
                               last_coord),
                         alpha=1,
                         facecolor = curr_color,
                         edgecolor=stroke if stroke else curr_color,
                         linewidth=self.half_width/10. if stroke else None)

class HexagonPrimitive(object):
    def __init__(self, x, y, width, color):
        self.half_width = width/2.
        self.h = SQRT_OF_THREE*self.half_width/3.
        self.x = x
        self.y = y
        self.color = color

    def draw(self, new_color=None, stroke=None):
        w = self.half_width
        h = self.h
        d_x = self.x
        d_y = self.y
        curr_color = new_color if new_color else self.color
        return Polygon(((d_x+w, d_y+h),
                        (d_x, d_y+2*h),
                        (d_x-w, d_y+h),
                        (d_x-w, d_y-h),
                        (d_x, d_y-2*h),
                        (d_x+w, d_y-h)),
                         alpha=1,
                         facecolor = curr_color,
                         edgecolor=stroke if stroke else curr_color,
                         linewidth=w/20. if stroke else None)


class Wavescape(object):
    '''
    This class represent an object that holds the attributes 
    and methods needed to effectively draw the wavescapes plot.

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
        
    subparts_highlighted: array of tuples of int, optional
        list of subsection that needs to be drawn with black outlines on the wavescape. Units
        are expressed in number of analysis windows. For example, if a musical piece has a 4/4 
        time signature, an analysis window of 1 quarter note and a total of 10 bars, the
        value [[20,28],[32,36]] for 'subparts_highlighted' will draw black outlines on th region
        of the wavescape corresponding to bars 5 until 7 and 8 until 9.
    '''
    #Formula derived with a bit of algebra in order to determine the height of the wavescape hexagon plot 
    #based on the just the given plot's width (wi) and the number of layer (n). The SQRT_OF_THREE*wi was broadcasted
    #to the two parts of the addition to mitigate the numeric error caused by the division by 6 times the number
    #of layer (n).
    HEXAGON_PLOT_HEIGHT = lambda wi, n: (SQRT_OF_THREE*wi)*(0.5) + ((SQRT_OF_THREE/6.)*(wi/n))
    
    #constants 
    #fun fact that would please anyone with OCD: all drawing primitives' name have the same amount of letters.
    DIAMOND_STR = 'diamond'
    RHOMBUS_STR = 'rhombus'
    HEXAGON_STR = 'hexagon'
    
    def __init__(self, utm, pixel_width, drawing_primitive='rhombus', subparts_highlighted=None):
        self.utm = utm
        self.width = pixel_width
        self.drawing_primitive = drawing_primitive
        
        mat_dim, mat_dim_other_axis, mat_depth = utm.shape
        if mat_dim != mat_dim_other_axis:
            raise Exception("The upper triangle matrix is not a square matrix")
        if mat_dim > self.width/2:
            raise Exception("The number of elements to be drawn exceeds the wavescape's resolution.(%d elements out of %d allowed by the resolution) Increase the width of the plot to solve this issue" % (mat_dim, self.width/2))
        if (mat_depth < 3 or mat_depth > 4):
            raise Exception("The upper triangle matrix given as argument does not hold either RGB or RGBA values")
        self.mat_dim = mat_dim
        
        self.subparts = subparts_highlighted
        
        #building a matrix with None to hold the element object for drawing them later.
        self.matrix_primitive = np.full((mat_dim, mat_dim), None, object)
        
        self.height = compute_height(self.width, mat_dim, drawing_primitive)
        if drawing_primitive == self.HEXAGON_STR:
            self.generate_hexagons(subparts_highlighted)
        elif drawing_primitive == self.RHOMBUS_STR or drawing_primitive == self.DIAMOND_STR:
            self.generate_diamonds(subparts_highlighted)
        else:
            raise Exception('Unkown drawing primitive: %s'%drawing_primitive)
            
    def generate_hightlights(self, unit_width):
        '''
        Helper method, is called by the other helper functions 'generate_diamonds/hexagons'. 
        Take care of generating the drawing primitive corresponding to the 
        highlights given as arguments to the constructor of the Wavescape class. 
        '''
        triangles = []
        for tup in self.subparts:
            lo = min(tup)
            hi = max(tup)
            if lo == hi:
                raise Exception('Highlight\'s start index (%s) should not be equal to its end index'%(str(lo)))
            if lo > self.mat_dim or hi > self.mat_dim:
                raise Exception('Subpart highlights\' indices cannot be above the number of element at the base of the wavescape (%d)'%self.mat_dim)
            tri_width = (hi-lo) * unit_width
            tri_height = compute_height(tri_width, hi-lo, self.drawing_primitive)
            xl = (lo-.5)*unit_width - self.width/2.
            yb = -self.height/2.
            xr = (hi-.5)*unit_width - self.width/2.
            yt = tri_height-self.height/2. 
            xt = (lo+hi-1)/2.*unit_width - self.width/2.
            triangles.append(Polygon(((xl, yb),
                               (xt, yt),
                               (xr, yb)),
                         alpha=1,
                         facecolor = None,
                         fill = None,
                         linewidth=1))
        self.subparts = triangles
                
    
    def generate_hexagons(self, subparts_highlighted):
        '''
        Helper method, is called by the constructor of the class. 
        This method takes care of generating the Hexagon drawing primitives in case such
        drawing primitive was chosen. One matplotlib.patches.Polygon is generated per element 
        of the plot. The draw method takes care of drawing those patches on the final figure.
        '''
        hexagon_width = self.width/float(self.mat_dim)
        hexagon_height = 2*SQRT_OF_THREE*hexagon_width/3.
        half_width_shift = self.width/2.
        half_height_shift = self.height/2.
        
        for y in range(self.mat_dim):
            for x in range(y, self.mat_dim):
                curr_color = rgb_to_hex(rgba_to_rgb(self.utm[y][x], background=(0xff,0xff,0xff)))
                #Useless to draw if there is nothing but blank to draw
                if curr_color != '#FFFFFF':
                    #classic x-axis placement taking into account the half width of the hexagon
                    d_x = hexagon_width*x
                    #Now shifting all of this to the left to go from utm placement to pyramid placement
                    d_x = d_x - hexagon_width*y/2.
                    #And finally shifting this to take into account drawSvg center placement I posed
                    d_x = d_x - half_width_shift
                    
                    d_y = hexagon_height/2.+(0.75*hexagon_height)*y
                    d_y = d_y - half_height_shift
                    
                    #self.matrix_primitive[y][x] = Hexagon(d_x, d_y, hexagon_width, curr_color)
                    self.matrix_primitive[y][x] = HexagonPrimitive(d_x, d_y, hexagon_width, curr_color)
        
        if subparts_highlighted:
            self.generate_hightlights(hexagon_width)
        else:
            self.subparts = None
    
    def generate_diamonds(self, subparts_highlighted):
        '''
        Helper method, is called by the constructor of the class. 
        This method takes care of generating the Diamond drawing primitives in case such
        drawing primitive was chosen. One matplotlib.patches.Polygon is generated per element 
        of the plot. The draw method takes care of drawing those patches on the final figure.
        '''
        diamond_width = self.width/float(self.mat_dim)
        diamond_height = diamond_width*2 if self.drawing_primitive != 'rhombus' else diamond_width * SQRT_OF_THREE
        
        half_width_shift = self.width/2.
        half_height_shift = self.height/2.
        
        for y in range(self.mat_dim):
            for x in range(y, self.mat_dim):
                
                curr_color = rgb_to_hex(rgba_to_rgb(self.utm[y][x], background=(0xff,0xff,0xff)))
                #Useless to draw if there is nothing but blank to draw, duh.
                if curr_color != '#FFFFFF':
                    #classic x-axis placement taking into account the edge from the diamond 
                    d_x = diamond_width*x
                    #Now shifting all of this to the left to go from utm placement to pyramid placement
                    d_x = d_x - diamond_width*y/2.
                    #And finally shifting this to take into account drawSvg center placement I posed
                    d_x = d_x - half_width_shift
                    
                    d_y = diamond_height/2.*y
                    d_y = d_y - half_height_shift
                    self.matrix_primitive[y][x] = DiamondPrimitive(d_x, d_y, \
                                          diamond_width, diamond_height, curr_color, y == 0)
        
        if subparts_highlighted:
            self.generate_hightlights(diamond_width)
        else:
            self.subparts = None

    def draw(self, ax=None, dpi=96, plot_indicators = True, add_line = False, tick_ratio = None, start_offset=0, label=None):
        '''
        After being called on a properly initialised instance of a Wavescape object,
        this method draws the visual plot known as "wavescape" and generate a 
        matplotlib.pyplot figure of it. This means any of the method from this
        library can be used after this method has been called in order to
        save or alter the figure produced. 

        Parameters
        ----------
        ax: matplotlib figure, optional
            Default value is None.
        
        plot_indicators: bool, optional 
            indicates whether rounded indicators on the lateral edges of the plot need to 
            be drawn. A rounded indicator is drawn at each eight of the height of the plot
            Default value is True
            
        dpi: int, optional
            dot per inch (dpi) of the figure. N
            Default value is 96, which is normally the dpi on windows machine. The dpi 
            
        add_line: bool, optional
            indicates whether all element of the plot (single drawing primitives) need to be
            outlined with a black line.
            Default value is False.
            
            
        tick_ratio: int, optional
            Ratio of tick per elements of the lowest level in the wavescape. If tick_ratio has value 1,
            one horizontal axis tick will be drawn per element at the lowest hierarchical level of the wavescape, 
            if it has value 2, then a tick will be drawn each two elements. For the ticks to represent the bar numbers,
            a preexisting knowledge of the time signature of the musical piece is required. (if a piece is in 4/4,
            and the analysis window has a size of one quarter note, then tick_ratio needs to have value 4 for the
            ticks to correspond to bar numbers)
            Default value is None (meaning no ticks are drawn)

        Returns
        -------
        Nothing
        '''
        utm_w = self.matrix_primitive.shape[0]
        utm_h = self.matrix_primitive.shape[1]
        
        if self.matrix_primitive is None or utm_w < 1 or utm_h < 1:
            raise Exception("cannot draw when there is nothing to draw. Don't forget to generate diamonds in the correct mode before drawing.")
        
        if tick_ratio: 
            
            if tick_ratio < 1 or type(tick_ratio) is not int:
                raise Exception("Tick ratio must be an integer greater or equal to 1")
            
            #argument start_offseet is only meaningless if there is tick ratio involved in the plotting
            if type(start_offset) is not int or start_offset < 0 or start_offset > tick_ratio:
                raise Exception("Stat offset needs to be a positive integer that is smaller or equal to the tick ratio")
        
        height = self.height
        width = self.width

        
        black_stroke_or_none = 'black' if add_line else None
        if not ax:
            fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
            ax = fig.add_subplot(111, aspect='equal')
        primitive_half_width = None
        

        for y in range(self.matrix_primitive.shape[0]):
            for x in range(y, self.matrix_primitive.shape[1]):
                element = self.matrix_primitive[y][x]
                if x == 1 and x == y:
                    primitive_half_width = element.half_width
                ax.add_patch(element.draw(stroke=black_stroke_or_none))
                             

        if plot_indicators:
            ind_width = width if self.drawing_primitive != self.HEXAGON_STR else width + 2
            mid_size = int(self.width / 40.)
            eigth_size = int(mid_size /4.)
            quart_size = eigth_size * 3

            white_fill = (1, 1, 1, 0)
            middle_gray= (.398, .398, .398, 1)

            params = [
                {'size': mid_size,   'facecolor': white_fill, 'edgecolor': 'black' },
                {'size': quart_size, 'facecolor': white_fill, 'edgecolor': middle_gray},
                {'size': eigth_size, 'facecolor': middle_gray,'edgecolor': middle_gray}
            ]

            stroke_width = int(self.width / 1000.)+1

            
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
                        ax.add_patch(Circle((i*x*width-primitive_half_width, y*height), radius=p['size'], facecolor=p['facecolor'], \
                                                  edgecolor=p['edgecolor'], linewidth=stroke_width))

        plt.autoscale(enable = True)
        
        labelsize = self.width/30.
        
        if tick_ratio:
            indiv_w = self.width/utm_w
            scale_x = indiv_w * tick_ratio
            ticks_x = FuncFormatter(lambda x, pos: '{0:g}'.format(math.ceil((x+ self.width/2.)/scale_x) + (1 if start_offset == 0 else 0)))
            
            ax.tick_params(which='major', length=self.width/50., labelsize=labelsize)
            ax.tick_params(which='minor', length=self.width/100.)
            
            ax.xaxis.set_major_formatter(ticks_x)
            number_of_ticks = self.width/scale_x
            eight_major_tick_base = scale_x*round(number_of_ticks/8.)
            ax.xaxis.set_major_locator(IndexLocator(base=eight_major_tick_base, offset=start_offset*indiv_w))
            
            #display minor indicators
            ax.xaxis.set_minor_locator(IndexLocator(base=scale_x, offset=start_offset*indiv_w))
            
            #make all the other border invisible
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plt.yticks([])
        else:
            plt.axis('off')
            
        if self.subparts:
            for pat in self.subparts:
                ax.add_patch(pat)

        #remove top and bottom margins 
        if label:
            x_pos = -self.width/2. + self.width/10.
            y_pos = self.height/2. - self.width/10.
            ax.annotate(label, (x_pos, y_pos), size=labelsize, annotation_clip=False, horizontalalignment='left', verticalalignment='top')
        ax.set_ylim(bottom=-self.height/2., top=self.height/2.)
        ax.set_xlim(left=-self.width/2.-primitive_half_width, right=self.width/2.-primitive_half_width)
        plt.tight_layout()

def generate_all_wavescapes(filepath,individual_width, save_label=None,\
                            aw_size=1, remove_percussions=True, trim_extremities=True,\
                            dpi=96, drawing_primitive=Wavescape.RHOMBUS_STR,\
                            tick_ratio=1, start_offset=0,\
                            plot_indicators=True, add_line=False):

    '''
    Generates the wavescapes for all six unique Fourier coefficients given
    the path of a musical piece in MIDI or WAV format. 
    Can output all 6 coefficients in a single figure, or output and save
    each coefficient separately.
    
    For small analysis window's size, this function will take some time to
    render all the individual figures.

    Parameters
    ---------
    filepath: str
        path to the MIDI or WAV file that gets visualized.
    
    individual_width: int
        the width in pixel of each individual wavescapes. If no save label is provided,
        then the resulting plot holds all 6 plots and consequently has 3*individual_width
        as width, and a height of two individual wavescapes (the hieght of a wavescape is
        dependent on the width and drawing primitive used)
    
    save_label: str, optional
        The prefix of the filepath to save each individual plot. If it has the (default)
        value of `None`, then the function produces all six plots into a single 3 by 2 figure
        and don't save it in PNG format (but this can be easily achieved by calling the "saveFig" 
        function of matplotlib.pyplot after this one)
        The path can be absolute or relative, however, it should not hold any file extensions at the end, 
        as it is generated by this function.
        For example, if the value "bach" is given for this parameter, then the following files will be created:
        bach1.png, bach2.png, bach3.png, bach4.png, bach5.png and bach6.png 
        Each number preceding the PNG extension indicates which coefficient is vizualized in the file.
        Default value is None.
    
    aw_size: int, optional
        see the doc of 'produce_pitch_class_matrix_from_filename' for information on this parameter.
        Default value is 1
        
    remove_percussions: boolean, optional
        see the doc of 'produce_pitch_class_matrix_from_filename' for information on this parameter.
        Default value is True
        
    trim_extremities: boolean, optional
        see the doc of 'produce_pitch_class_matrix_from_filename' for information on this parameter.
        Default value is True
        
    dpi: int, optional
        see the doc of the constructor of the class 'Wavescape' for information on this parameter.
        Default value is 96 (most common dpi values for computers' screen)
    
    drawing_primitive: str, optional
        see the doc of the constructor of the class 'Wavescape' for information on this parameter.
        Default value is Wavescape.RHOMBUS_STR (i.e. 'rhombus')
    
    tick_ratio: int, optional 
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is 1
    
    start_offset: int, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is 0
    
    plot_indicators: boolean, optional 
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is True
    
    add_line: boolean, optional
        see the doc the 'draw' method from the class 'Wavescape' for information on this parameter.
        Default value is False
    
    '''

    pc_mat = produce_pitch_class_matrix_from_filename(filepath, aw_size=aw_size)
    fourier_mat = apply_dft_to_pitch_class_matrix(pc_mat)
    
    total_width = (3.1*individual_width)/dpi
    total_height = (2.1*compute_height(individual_width, fourier_mat.shape[0], drawing_primitive))/dpi
    if not save_label:
        fig = plt.figure(figsize=(total_width, total_height), dpi=dpi)
    
    for i in range(1, 7):
        color_utm = complex_utm_to_ws_utm(fourier_mat, coeff=i)
        w = Wavescape(color_utm, pixel_width=individual_width, drawing_primitive=drawing_primitive)
        if save_label:
            w.draw(plot_indicators=plot_indicators, add_line=add_line,\
               tick_ratio=tick_ratio, start_offset=start_offset)
            plt.tight_layout()
            plt.savefig(save_label+str(i)+'.png', bbox_inches='tight')
        else:
            ax = fig.add_subplot(2, 3, i)
            w.draw(ax=ax, plot_indicators=plot_indicators, add_line=add_line,\
               tick_ratio=tick_ratio,start_offset=start_offset,  label=coeff_nbr_to_label(i)+' coeff.')
            
    plt.tight_layout()

def legend_decomposition(pcv_dict, width = 13, single_img_coeff = None):
    '''
    Draw the circle color space defined by the color mapping used in wavescapes.
    Given a dict of labels/pitch-class vector, and list of coefficient to visualize,
    this function will plot the position of each of the PCV on the coefficient selected.
    This function helps visualising which color of the wavescape correspond to which musical 
    structure with respect to the coefficient number.
    
    Parameters
    ----------
    
    pcv_dict: dict, type of key is str, type of value is an array of array dimension (2,N) (0<= N, <=12)
        defines the label and pitch-class vector to be drawn, as well as the list of coefficients on which
        the pitch-class vector position needs to be drawn. For example, consider this dict is given to the
        function:
        {'CMaj':[[1,0,1,0,1,1,0,1,0,1,0,1], [5]],
         'Daug':[[0,0,1,0,0,0,1,0,0,0,1,0], [3,6]],
         'E': [0,0,0,0,1,0,0,0,0,0,0,0], [0]}
         The position of the C Major diatonic scale will be drawn on the color space of the fifth coefficient,
         while the position of the D augmented triad will be drawn on the color space of both the third and
         sixth coefficient. Finally, the value 0 associated to the single pitch PCV 'E' indicates its position
         will be drawn on all of the 6 coefficients. The label support LateX math mode.
    
    width: int, optional
        plot's width in inches.
        Default value is 13.
        
    single_img_coeff: int, optional
        Indicates which coefficient's color space will be drawn. If no number or "None" is provided for the value
        of this parameter, then the resulting plots will feature all 6 color space, one per coefficient. The coefficient
        number contain in the dict 'pcv_dict' still apply if a single coefficient is selected with this parameter.
        Default value is None.
        
    '''
    phivals = np.arange(0, 2*np.pi, 0.01)
    mu_step = .025
    muvals = np.arange(0, 1. + mu_step, mu_step)
    
    #powerset of all phis and mus.
    cartesian_polar = np.array(np.meshgrid(phivals, muvals)).T.reshape(-1, 2)
    
    #generating the color corresponding to each point.
    color_arr = []
    for phi, mu in cartesian_polar:
        hexa = rgb_to_hex(circular_hue(phi, magnitude=mu, opacity_mapping=True))
        color_arr.append(hexa)
        
    xvals = cartesian_polar[:,0]
    yvals = cartesian_polar[:,1]

    norm = mpl.colors.Normalize(0.0, 2*np.pi)
    fig = plt.figure(figsize= (width,width) if single_img_coeff else (width, 8*width/5) )
    
    def single_circle(ax, i, pcv_dict, marker_width, display_title=True):
        label_size = (marker_width/10.)
        ax.scatter(xvals, yvals, c=color_arr, s=marker_width, norm=norm, linewidths=1, marker='.')
        if display_title:
            ax.set_title(coeff_nbr_to_label(i)+' coefficient', fontdict={'fontsize': label_size+6}, y=1.08)
        for k,v in pcv_dict.items():
            for coeff in v[1]:
                if coeff == i or coeff == 0:
                    comp = np.fft.fft(v[0])
                    angle = np.angle(comp[i])
                    magn = np.abs(comp[i])/np.abs(comp[0])
                    ax.scatter(angle, magn, s=marker_width, facecolors='none', edgecolors='#777777')
                    pos_magn = np.abs(magn-0.125)
                    ax.annotate(k, (angle, pos_magn), size=(marker_width/10.)+2, annotation_clip=False, horizontalalignment='center', verticalalignment='center')
        
        ax.tick_params(axis='both', which='major', labelsize=(marker_width/10.)+6)
        ax.tick_params(axis='both', which='minor', labelsize=(marker_width/10.)+4)
        ax.set_xticklabels(['$0$', '', '$\pi/2$', '', '$\pi$', '', '$3\pi/2$', ''])
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        ax.xaxis.grid(False)
    
    if single_img_coeff:
        ax = plt.subplot(1, 1, 1, polar=True)
        single_circle(ax=ax, i=single_img_coeff, pcv_dict=pcv_dict, marker_width=20*width, display_title=False)
    else:
        for i in range(1, 7):
            ax = fig.add_subplot(3, 2, i, polar=True)
            single_circle(ax=ax, i=i, pcv_dict= pcv_dict, marker_width=10*width)
        plt.tight_layout() #needs to be before subplot_adjust, otherwise subplot_adjust is useless.
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.3)
        
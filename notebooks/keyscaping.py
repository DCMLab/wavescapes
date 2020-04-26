import numpy as np
import music21
import math
import matplotlib.pyplot as plt
import drawSvg as draw
from librosa.feature import chroma_stft
from scipy.io import wavfile
from IPython.display import display

from scipy.misc import toimage
from PIL import Image, ImageOps, ImageDraw
from scipy.stats import describe


#########################################################
# Rough structure of the code
#########################################################
# This code consist in basically 4 big parts: 
# pertinent constants and metadata,
# pipelining code, keyscaping code, and drawing code. 
#
# The first part are relevant informations and dict for the 
# code, all related to key and pitch classes.
#
# The second part is simply three function 
# transforming its input into and intermediate 
# representation usefule for the next function,the end 
# result of pipeling the three functions gives a keyscape.
#
# The third part are functions that transform the midi
# piece into intermediate data representation used in order
# eventually produce a keyscape
# 
# Finally the last part is the code that will be useful to
# draw the actual keyscape from a similar (in terms of 
# information) upper triangle matrix
# 
# For an example of execution, take a look a the main at
# the bottom of the code.



#########################################################
# Metadata & Constants & other general things. 
#########################################################


twelve_tones_major_key_names = ['C','Db','D','Eb','E','F','Gb','G','Ab','A','Bb','B']
twelve_tones_minor_key_names = ['c','c#','d','d#','e','f','f#','g','g#','a','bb','b']

#Source: Temperley, D. (1999). What's Key for Key? The Krumhansl-Schmuckler Key-Finding Algorithm Reconsidered. Music Perception: An Interdisciplinary Journal, Vol. 17, No. 1 (Fall, 1999), pp. 65-100
krumhansl_maj = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
krumhansl_min = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

temperley_maj = [5.0, 2.0, 3.5, 2.0, 4.5, 4.0, 2.0, 4.5, 2.0, 3.5, 1.5, 4.0]
temperley_min = [5.0, 2.0, 3.5, 4.5, 2.0, 4.0, 2.0, 4.5, 3.5, 3.0, 1.5, 4.0]

#special case in order to have silence being represented as white on the keyscapes
white_value = 42

def rgba_to_rgb(to_convert, background):
    if len(to_convert) == 3:
        return to_convert #no point converting something that is already in RGB
    if len(to_convert) != 4:
        raise Exception('Incorrect format for the value to be converted, should have length of 4')
    if len(background) != 3:
        raise Exception('Incorrect format for the value background, should have length of 3 (no alpha channel for this one)')
    alpha = float(to_convert[3])/255.0
    return [int((1 - alpha) * background[i] + alpha * to_convert[i]) for i in range(len(background))]

def circular_hue(angle, opacity=0xff, output_rgb=True):
    
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
    value = (stand(red(angle)), stand(green(angle)), stand(blue(angle)), opacity)
    if output_rgb:
        value = rgba_to_rgb(value, background=(0xff,0xff,0xff))
    return value

def nth_coeff_phase(input_vec, n_coeff):
    assert(n_coeff <= len(input_vec))
    if not np.any(input_vec):
        return white_value
    dft_nth_coeff = np.fft.fft(input_vec)[n_coeff]
    return np.angle(dft_nth_coeff)

def nth_coeff_magn(input_vec, n_coeff):
    assert(n_coeff <= len(input_vec))
    if not np.any(input_vec):
        return white_value
    dft_nth_coeff = np.fft.fft(input_vec)[n_coeff]
    return np.abs(dft_nth_coeff)

def nth_coeff_magn_and_phase(input_vec, n_coeff, use_white_value=True):
    assert(n_coeff <= len(input_vec))
    if use_white_value and not np.any(input_vec):
        return white_value
    dft_nth_coeff = np.fft.fft(input_vec)[n_coeff]
    return (np.abs(dft_nth_coeff), np.angle(dft_nth_coeff))

temperley_minor_keys = [np.roll(temperley_min, i) for i in range(len(temperley_min))]
temperley_major_keys = [np.roll(temperley_maj, i) for i in range(len(temperley_maj))]

temperley_major_angle_dict = {k: nth_coeff_phase(v, 5) for k, v in zip(twelve_tones_major_key_names, temperley_major_keys)}
temperley_minor_angle_dict = {k+'m': nth_coeff_phase(v, 5) for k, v in zip(twelve_tones_minor_key_names, temperley_minor_keys)}


temperley_major_color_dict = {k: circular_hue(v) for k,v in temperley_major_angle_dict.items()}
#half the opacity 
temperley_minor_color_dict = {k: circular_hue(v, opacity=0x77) for k,v in temperley_minor_angle_dict.items()}

tempereley_label_profile_major = dict(zip(twelve_tones_major_key_names, temperley_major_keys))
#adding small m so that when drawing the circle of pitches on the hue the minor keys will be drawn on the inner circle.
tempereley_label_profile_minor = dict(zip([s+'m' for s in twelve_tones_minor_key_names], temperley_minor_keys))


twelve_tones_vector_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#','A', 'A#', 'B']
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

#color values were derived by mapping a CIEL*a*b square color space on a paralleogram which is a  
# subsection of the 12 tones on the tonnetz
tonnetz_pitch_to_rgb_major = {
    'C': (0,255,0),
    'C#': (7,88,255),
    'D': (6,191,217),
    'D#': (235, 33, 93),
    'E': (249, 173,2),
    'F': (26, 4, 250),
    'F#': (66, 136, 195),
    'G': (0, 246, 109),
    'G#': (250,4,5),
    'A': (7,130,254),
    'A#': (255, 0, 180),
    'B': (195, 171, 97)
}

tonnetz_pitch_to_rgb_minor = {
    'C': (172,85,102),
    'C#': (166,139,97),
    'D': (69,83,171),
    'D#': (87, 122, 150),
    'E': (98, 166,115),
    'F': (171, 85, 81),
    'F#': (83, 122, 170),
    'G': (171, 86, 142),
    'G#': (144,138,108),
    'A': (102,173,95),
    'A#': (74, 107, 170),
    'B': (91, 144, 156 )
}

tonnetz_index_rgb_dict_minor = {index: tonnetz_pitch_to_rgb_minor[pitch] for pitch,index in pitch_index_dict.items()}

tonnetz_index_rgb_dict_major = {index: tonnetz_pitch_to_rgb_major[pitch] for pitch,index in pitch_index_dict.items()}


#########################################################
# Pipelining functions
#########################################################



def produce_pitch_offset_list_from_midi_filename(midi_filename):
    midi_stream = music21.converter.parse(midi_filename)
    return recursively_map_offset(midi_stream)


def produce_ks_as_utm_format_from_pitch_offset_list(pitch_offset_list):

    temperley_color_mapping = lambda pitch_class: pitch_to_color_template_matching(pitch_class, temperley_maj, temperley_min, temperley_major_color_dict, temperley_minor_color_dict)


    return generalized_keyscape(pitch_class_set_vector_from_pitch_offset_list(pitch_offset_list), temperley_color_mapping)


def generate_keyscape_from_ks_utm(ks_utm, keyscape_width):
    ks = Keyscape(ks_utm, keyscape_width)
    ks.generate_diamonds()
    return (ks.draw(),display_key_colors(temperley_major_color_dict, temperley_minor_color_dict, keyscape_width))


#########################################################
# Keyscaping functions
#########################################################

def recursively_map_offset(midi_stream, flatten=True, only_note_name=True):
    '''
    This function will recursively walk through the Midi stream's elements, 
    and whenever it encounters a Note or a chord, it will append its rhytmic 
    data to it and then store the resulting in an array. Returns the
    aforementionned array.
    The rhytmic data is expressed as a tuple of the offset of the beginning of the note
    and the offset of the end of the note. 
    
    Params: 
    midi stream: the MIDI stream containing all the relevant infos
    flatten: Boolean, indicating whether or note the chords elements 
            need to be flattened into singles notes.
    only_note_name: Boolean, indicating whether the notes need to be 
                    converted from music21 object with octave indication
                    to only a string indicating the pitch.
    '''
    
    if flatten and not only_note_name:
        raise Exception("Cannot retrieve the pitch name on Chord structure")
    res = []
    for elem in midi_stream.recurse():
        if isinstance(elem, music21.note.Note):
            start_offset = elem.getOffsetInHierarchy(midi_stream)
            res.append((elem.name if only_note_name else elem, (start_offset, start_offset+elem.duration.quarterLength)))
        elif isinstance(elem, music21.chord.Chord):
            start_offset = elem.getOffsetInHierarchy(midi_stream)
            if flatten:
                res = res + list(map(lambda r: (r.name if only_note_name else r , (start_offset, start_offset+elem.duration.quarterLength)), elem.pitches))
            else:
                res.append([elem, offset])
    return res


def only_keep_pitches_in_boundaries(pitch_offset_list, beat1_offset, beat2_offset): 
    return list(filter(lambda n: n[1][1] >= beat1_offset and n[1][0] <= beat2_offset, pitch_offset_list))


def slice_according_to_beat(pitch_offset_list, beat1_offset, beat2_offset):
    #the beat offset must be expressed as relation of quarter note. 
    #Taken are all beat which at least END AFTER the beat1, and START BEFORE the beat2
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



def get_max_beat(pitch_offset_list):
    return math.ceil(max(list(map(lambda r: r[1][1], pitch_offset_list))))


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


def scalar_product(input_vec, key_prof, shift_idx):
    # compute the scalar product between a pitch class (input_vec),
    # and a key profile starting at C (key_prof). The argument "shift_idx"
    # indicates at which pitch should the scalar product begin according to the 
    # key profile.
    acc = 0
    pitch_class_size = len(key_prof)
    for i in range(pitch_class_size):
        acc += input_vec[i]*key_prof[(i-shift_idx)%pitch_class_size]
    return acc


def select_best_key_sp(input_vec, key_prof_maj, key_prof_min):
    #returns a tuple of type int and boolean: (key_index, isMajor)
    
    result_array_maj = np.array([ scalar_product(input_vec, key_prof_maj, i) for i in range(len(key_prof_maj))])
    result_array_min = np.array([ scalar_product(input_vec, key_prof_min, i) for i in range(len(key_prof_min))])
    key_idx = np.argmax(np.vstack([result_array_maj, result_array_min]))
    #np.argmax return only a singular index, regardless of the array shape, meaning that above 11, that
    #that means the key was identified in the second row, i.e. the minor key.
    return (key_idx % 12, key_idx < 12)    


def pitch_to_color_template_matching(pitch_class_vector, major_profile, minor_profile, major_colors, minor_colors):
    if not np.any(pitch_class_vector):
        #this means the vector only has 0 values.
        return (255,255,255,255)
    key, is_major = select_best_key_sp(pitch_class_vector, major_profile, minor_profile)
    pitch_to_rgb = major_colors if is_major else minor_colors
    indexes = [i for i in range(len(pitch_to_rgb))]
    index_to_rgb = dict(zip(indexes, pitch_to_rgb.values()))
    return index_to_rgb[key]



def pitch_class_set_vector_from_pitch_offset_list(pitch_offset_array, chunk_number=0.0): #the analysis window size (aw_size) is expressed in terms of number of quarter.
    max_beat = get_max_beat(pitch_offset_array)
    chunk_number = chunk_number if chunk_number > 0. else max_beat
    aw_size = float(max_beat)/chunk_number
    res_vector = np.full((chunk_number, 12), 0.0, np.float32)

    for b in range(math.ceil(max_beat/aw_size)):
        start_beat = b*aw_size
        stop_beat = (b+1)*aw_size
        analysis_window = slice_according_to_beat(pitch_offset_array, start_beat, stop_beat)
        pitch_class_vec = sum_into_pitch_class_vector(analysis_window, start_beat, stop_beat)
        res_vector[b] = pitch_class_vec
    
    return res_vector

# trim the input array so that no empty vectors 
def trim_pcs_array(pcvs):
    start = 0
    while not np.any(pcvs[start]):
        start += 1
    end = len(pcvs) - 1
    while not np.any(pcvs[end]):
        end -= 1
    return pcvs[start:end+1]

def color_mapping_phase_and_magnitude_with_opacity(input_vec, fourier_coeff, normalizing_function = sum):
    MnP = nth_coeff_magn_and_phase(input_vec, fourier_coeff)
    vector_weight = sum(input_vec)
    if vector_weight == 0 or vector_weight < MnP[0]:
        #if silences are present, we get a norm_magn that can get to infinity without that! 
        return (0,0,0,0)
    magnitude = MnP[0]
    phase = MnP[1]
    norm_magn = magnitude/normalizing_function(input_vec)
    opacity_as_hex = int(norm_magn*float(0xff))
    return circular_hue(phase, opacity=opacity_as_hex, output_rgb=False)
    
'''
pcs_array is an 2 dimensional array, with the second dimension
always being 12 (number of pitches in one pitch class set) and the first
being the number of sample used to produce a keyscape.

The result will be a UTM with each value being a RGBA value. 
'''
def generalized_keyscape(pcs_array, pitch_class_to_color):

    #empty PCV frame at the beginning and end of the score serve no purpose other than taking place and change the shape of the keyscape.
    pcs_array = trim_pcs_array(pcs_array)
    
    #getting the max possible 'beat' from the piece
    max_beat = len(pcs_array)
    
    res_matrix = np.full((max_beat,max_beat,4), 0xFF, np.uint8)
    
    memoization_matrix = np.full((max_beat, max_beat, 12), 0., np.float32)
    memoization_matrix[0] = pcs_array

    #need to do independantly the filling of the first row of the matrix
    for b in range(max_beat):
        color_value = pitch_class_to_color(pcs_array[b])
        if len(color_value) == 3:
            #we got RGB value, we will convert them to RGBA value with the most brightness
            color_value = (color_value[0],color_value[1],color_value[2], 0xFF)
        res_matrix[0][b] = color_value
    
    for i in range(1, max_beat):
        
        curr_row = res_matrix[i]
        previous_mem_row = memoization_matrix[i-1]
        curr_mem_row = memoization_matrix[i]
        for j in range(0, max_beat-i):
            #To avoid adding twice the common pitch classes between the two point in the matrix
            #below, we only take the previous summed value and add to it only one new beat, 
            #hence the use of the 0th row of the matrix
            pitch_class_vec = memoization_matrix[0][i+j] + previous_mem_row[i+j-1]
            color_value = pitch_class_to_color(pitch_class_vec)
            if len(color_value) == 3:
                #we got RGB value, we will convert them to RGBA value with the most brightness
                color_value = (color_value[0],color_value[1],color_value[2], 0xFF)
            curr_row[j+i] = color_value
            curr_mem_row[i+j] = pitch_class_vec
    
    return res_matrix



def angle_magnitude_utms_production(pcs_array, fourier_coeff):
    
    #getting the max possible 'beat' from the piece
    max_beat = len(pcs_array)
    
    angle_matrix = np.full((max_beat,max_beat), 0., np.float32)
    magn_matrix = np.full((max_beat,max_beat), 0., np.float32)
    
    memoization_matrix = np.full((max_beat, max_beat, 12), 0., np.float32)
    memoization_matrix[0] = pcs_array

    #need to do independantly the filling of the first row of the matrix
    for b in range(max_beat):
        pcs_elem = pcs_array[b]
        magn, angle = nth_coeff_magn_and_phase(pcs_elem, fourier_coeff, use_white_value=False)
        angle_matrix[0][b] = angle
        magn_matrix[0][b] = magn if magn ==0 else magn/float(sum(pcs_elem))
    
    for i in range(1, max_beat):
        
        curr_row_angle = angle_matrix[i]
        curr_row_magn = magn_matrix[i]
        previous_mem_row = memoization_matrix[i-1]
        curr_mem_row = memoization_matrix[i]
        for j in range(0, max_beat-i):
            #To avoid adding twice the common pitch classes between the two point in the matrix
            #below, we only take the previous summed value and add to it only one new beat, 
            #hence the use of the 0th row of the matrix
            pitch_class_vec = memoization_matrix[0][i+j] + previous_mem_row[i+j-1]
            magn, angle = nth_coeff_magn_and_phase(pitch_class_vec, fourier_coeff, use_white_value=False)
            
            angle_matrix[i][j+i] = angle
            #careful with that, if the array is only zero with just a small index with a value close to zero but not zero, this will generate huge values. 
            magn_matrix[i][j+i] = 0.0 if magn == 0.0 else magn/sum(pitch_class_vec)
            
            curr_mem_row[i+j] = pitch_class_vec
    
    return (angle_matrix, magn_matrix)

def float_utm_to_color(utm, value_to_color, output_dim, zero_value):
    assert(output_dim == len(zero_value) if output_dim > 1 else True) 
    utm_dim1, utm_dim2 = np.shape(utm)
    assert(utm_dim1 == utm_dim2)
    
    res_matrix = np.full((utm_dim1,utm_dim1,output_dim), [0xFF]*output_dim, np.uint8)
    for i in range(utm_dim1):
        for j in range(utm_dim2):
                res_matrix[i][j] = zero_value if j < i else value_to_color(utm[i][j])
    return res_matrix


def center_of_mass(utm):
    #the tuples returned has its first value corresponding to the height of the center of mass (i being the line index) and 
    #the second value is the width of the center of mass (j being the column index).
    magn_sum = np.sum(utm)
    N = np.shape(utm)[0]
    curr_sum = np.array([0.,0.])
    for i in range(N):
        for j in range(i, N):
            particule_coord = np.array([i, j]) + (1,1) #0 indexing won't work properly with the CM equation.
            curr_sum += particule_coord*utm[i][j]
    return (curr_sum/magn_sum) - (1,1)#compensating the 0 indexing fix made two lines above.


def center_of_mass_height(utm):
    magn_sum = np.sum(utm)
    N = np.shape(utm)[0]
    curr_sum = 0.
    for i in range(N):
        for j in range(i, N):
            curr_sum += (i+1)*utm[i][j]
    return (curr_sum/magn_sum) - 1.

def magn_to_opacity(magn, test_boundaries=True):
    if test_boundaries:
        if magn >= 1.0:
            return 0xff
        elif magn <= 0.0:
            return 0
    return int(float(0xff)*magn)

opacity_to_rgb = lambda v: [magn_to_opacity(v)]*3
opaque_circular_hue = lambda v: circular_hue(v, output_rgb=False)

def apply_alpha_channel(utm, alpha_utm):
    utm = utm.copy()
    shapes = np.shape(utm)
    for i in range(shapes[0]):
        for j in range(shapes[1]):
            utm[i][j][3] = alpha_utm[i][j]
    return utm

def increase(to_increase, max_val, increment):
    new_val = to_increase + increment
    if new_val > max_val:
        return max_val
    else:
        return new_val
    
def alter_alpha_channel(utm, increment):
    shapes = np.shape(utm)
    for i in range(shapes[0]):
        for j in range(shapes[1]):
            utm[i][j][3] = increase(utm[i][j][3], 0xff, increment)
    return utm

def fetch_utm_values(mask_utm, is_float=True):
    values_recorded = []
    size = np.shape(mask_utm)[0]
    for i in range(size):
        for j in range(size-i):
            values_recorded.append(mask_utm[i][j])
    return values_recorded

def max_scale_matrix(magnitude_matrix):
    mat_max = magnitude_matrix.max()
    return magnitude_matrix*(1.0/mat_max)


def median_scale_matrix(magnitude_matrix):
    median = np.median(magnitude_matrix)
    #new_median == 0.5
    #factor*old_median == new_median
    #factor == new_median/old_median = 0.5/old_median
    return magnitude_matrix*(0.5/median)


def median_half_scale_matrix(magnitude_matrix):
    median = np.median(magnitude_matrix)
    return magnitude_matrix*(0.25/median)

def mean_scale_matrix(magnitude_matrix):
    mean = magnitude_matrix.mean()
    return magnitude_matrix*(0.5/mean)

def void_magn_matrix(mat):
    mat[mat > 0] = 1.
    return mat

def rectify_magnitude_value(magn_val, coeff):
    #obtained experimentally by plotting the magnitude of all unique vectors (in terms of translation and inversion) of the powerset of 2 to 12. 
    coeff_range_dict = {
        1: [0.0, 1.0],
        2: [0.15, 0.9],
        3: [0.1, 0.8],
        4: [0.1, 0.75],
        5: [0.0, 1.0],
        6: [0.1, 0.7]
    }
    bounds = coeff_range_dict[coeff]
    input_start = bounds[0]
    input_end = bounds[1]
    output_start = 0.0
    output_end = 1.0
    if math.isclose(magn_val, output_start, rel_tol=1e-7, abs_tol=1e-7) or math.isclose(magn_val, output_end, rel_tol=1e-7, abs_tol=1e-7):
        return magn_val
    else:
        return output_start + ((output_end - output_start) / (input_end - input_start)) * (magn_val - input_start)
    

#########################################################
# Audio signal processing functions
#########################################################

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
def produce_chromagrams_from_audio_file(audio_filename, number_of_aw):
    audio_array, sample_ratio = audiowav_to_mono_float_array(audio_filename)
    hop_len = math.floor(len(audio_array)/number_of_aw)
    #so the result's shape is consistent with the one produced in the case of midi files.
    return np.transpose(chroma_stft(audio_array, sample_ratio, hop_length=hop_len))


#########################################################
# Drawing functions
#########################################################


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
 

def display_key_colors(maj_key_dict, min_key_dict=None, total_width=1000):
    if min_key_dict:
        assert(len(maj_key_dict.items()) == len(min_key_dict.items()))
    
    #taking one quarter of a square to distance each of them between them. meaning we have the total width that
    #needs to be equal to 12*square_width + 12*(1/4)*square = 15square_width
    number_of_square_width = len(maj_key_dict)
    number_of_square_width = number_of_square_width + number_of_square_width/4.
    square_width = total_width/number_of_square_width
    total_height = 5*square_width if min_key_dict else 3*square_width #4 square width + 4 quarters of a square width
    
    #origin is not the center cause it made mapping too complicate for my taste
    canvas = draw.Drawing(total_width, total_height)
    canvas.append(draw.Rectangle(0,0, total_width, total_height, fill='white'))
    
    maj_array = list(maj_key_dict.items())

    if min_key_dict:
        min_array = list(min_key_dict.items())
        
    
    for i in range(len(maj_array)):
        
        
        quarter_sw = square_width/4
        x_offset = (square_width + square_width/4)*i + square_width/8 #the 1/8 at the end if for centering
        
        maj_label, maj_rgb_value = maj_array[i]
        maj_color = rgb_to_hex(maj_rgb_value)
        canvas.append(draw.Text(maj_label, square_width/2,x_offset, quarter_sw , fill='black'))
        canvas.append(draw.Rectangle(x_offset, quarter_sw+square_width, square_width, square_width, fill=maj_color))
    
        if min_key_dict:
            min_label, min_rgb_value = min_array[i]
            min_color = rgb_to_hex(min_rgb_value)
            canvas.append(draw.Text(min_label, square_width/2 ,x_offset, 2*(square_width+quarter_sw), fill='black'))
            canvas.append(draw.Rectangle(x_offset, quarter_sw+3*square_width, square_width, square_width, fill=min_color))

    return canvas  

#In drawsvg, zou can change 'opacity' by adding two hex char at the end of the color. This acts as the alpha channel. 
def percentage_to_hex_opacity(percent, quarter_progression=True):
    max_value = 0xff
    #max_value = int(max_value)/
    curr_value = min(int((percent+0.01)*max_value), max_value)
    return '%02X'%curr_value 


def draw_utm_boundaries(im):
    #adding three to account for correct boundaries drawing.
    res_img = Image.new('RGBA', (im.size[0]+3, im.size[0]+3), (0xff, 0xff, 0xff, 0xff))
    res_img.paste(im, (2,1))
    
    draw = ImageDraw.Draw(res_img)
    
    draw.line((0, 0) + res_img.size, fill=(0,0,0,255))
    draw.line((0, 0, res_img.size[0]-1, 0), fill=(0,0,0,255))
    draw.line((res_img.size[0]-1, 1, res_img.size[0]-1, res_img.size[0]-1), fill=(0,0,0,255))

    del draw
    return res_img

def fast_plotting_hack(utm, add_boundaries=False):
    
    #puttin the utm with the lowest level on top, and the order of the musical sequence from left to right. 
    utm_img = toimage(utm)
    if add_boundaries:
        utm_img = draw_utm_boundaries(utm_img)
    im = ImageOps.mirror(ImageOps.flip(utm_img))
    
    width, height = im.size
    m = 0.5
    xshift = abs(m) * width
    new_width = width + int(round(xshift))
    #shifting the top dimension of the utm to the left so the utm now resembles a keyscape
    im = im.transform((new_width, height), Image.AFFINE,(1, m, -xshift if m > 0 else 0, 0, 1, 0), Image.BICUBIC)
    
    left = 0
    top = 0
    right = int(2.0*(im.width/3.0))+1 
    bottom = im.height
    
    # Cropped image of above dimension, returns a new image.
    return im.crop((left, top, right, bottom))



def print_utm_stats(mutm):
    print('Magnitude stats for coeff %d:\nMax:%lf\nMin:%lf\nMedian:%lf\nMean:%lf'\
          %(coefficient, mutm.max(), mutm.min(), np.median(mutm), mutm.mean()))

def utm_from_pcv(pcv, coefficient, print_stats=False):
    autm, mutm =  angle_magnitude_utms_production(pcv, coefficient)
    opaque_utm = float_utm_to_color(autm, opaque_circular_hue, 4, [0xff]*4)
    max_mask_utm = float_utm_to_color(max_scale_matrix(mutm), magn_to_opacity, 1, 0xff)
    if print_stats:
        print_utm_stats(mutm)

    return apply_alpha_channel(opaque_utm, max_mask_utm)

def slow_keyscape_boosted_magnitude_midi(midi_filename, coefficient, resolution):
    pol = produce_pitch_offset_list_from_midi_filename(midi_filename)
    midi_pcv = pitch_class_set_vector_from_pitch_offset_list(pol)
    max_full_utm = utm_from_pcv(midi_pcv, coefficient)
    print('Magnitude stats for coeff %d:\nMax:%lf\nMin:%lf\nMedian:%lf\nMean:%lf'\
          %(coefficient, mutm.max(), mutm.min(), np.median(mutm), mutm.mean()))

    ks = Keyscape(max_full_utm, resolution)
    ks.generate_diamonds()
    return ks.draw()


def fast_keyscape_boosted_magnitude_midi(midi_filename, coefficient):
    pol = produce_pitch_offset_list_from_midi_filename(midi_filename)
    midi_pcv = pitch_class_set_vector_from_pitch_offset_list(pol)
    max_full_utm = utm_from_pcv(midi_pcv, coefficient)
    return fast_plotting_hack(max_full_utm)

def fast_keyscape_boosted_magnitude(audio_filename, coefficient, resolution):
    wav_pcv = produce_chromagrams_from_audio_file(audio_filename, resolution)
    max_full_utm = utm_from_pcv(wav_pcv, coefficient)
    print('Magnitude stats for coeff %d:\nMax:%lf\nMin:%lf\nMedian:%lf\nMean:%lf'\
          %(coefficient, mutm.max(), mutm.min(), np.median(mutm), mutm.mean()))
    return fast_plotting_hack(max_full_utm)


class Diamond(object):
    def __init__(self, width, height, x, y, color):
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.color = color
        
    def draw(self, canvas, new_color=None, stroke=None):
        new_color = new_color if new_color else self.color
        stroke = stroke if stroke else self.color
        half_width = self.width/2
        half_height = self.height/2
        d_x = self.x
        d_y = self.y
        canvas.append(draw.Lines(d_x-half_width, d_y,
                    d_x, d_y+half_height,
                    d_x+half_width, d_y,
                    d_x, d_y-half_height,
            fill=new_color,
            stroke=stroke))

    

class Keyscape(object):
    def __init__(self, utm, width):
        self.utm = utm
        self.width = width
        self.matrix_diamonds = None
    
    def generate_diamonds(self):
        mat_dim, mat_dim_other_axis = self.utm.shape[0], self.utm.shape[1]
        if mat_dim != mat_dim_other_axis:
            raise Exception("The upper triangle matrix is not a square matrix")
            
        if mat_dim > self.width/2:
            raise Exception("The number of elements to be drawn exceeds the keyscape's resolution.(%d elements out of %d allowed by the resolution) Increase the width of the keyscape to solve this issue" % (mat_dim, self.width/2))
        diamond_width = self.width/mat_dim
        diamond_height = diamond_width*2
        #building a matrix with None to hold the diamonds object for drawing them later.
        self.matrix_diamonds = np.full((mat_dim, mat_dim), None, object)
        for y in range(mat_dim):
            for x in range(y, mat_dim):
                
                curr_color = rgb_to_hex(rgba_to_rgb(self.utm[y][x], background=(0xff,0xff,0xff)))
                #Useless to draw if there is nothing but blank to draw, duh.
                if curr_color != '#FFFFFF':
                    #classic x-axis placement taking into account the edge from the diamond 
                    d_x = diamond_width/2 + diamond_width*x
                    #Now shifting all of this to the left to go from utm placement to pyramid placement
                    d_x = d_x - diamond_width*y/2
                    #And finally shifting this to take into account drawSvg center placement I posed
                    d_x = d_x - self.width/2
                    
                    d_y = diamond_height/2*y
                    d_y = d_y - self.width/2
                    res_diamond = Diamond(diamond_width, diamond_height, d_x, d_y, curr_color)
                    self.matrix_diamonds[y][x] = res_diamond

    def draw(self, plot_indicators = True, add_line = False, add_bg=True):
            if self.matrix_diamonds is None or self.matrix_diamonds.shape[0] < 1 or self.matrix_diamonds.shape[1] < 1:
                raise Exception("cannot draw when there are nothing to draw. Don't forget to generate diamonds in the correct mode before drawing.")
            canvas = draw.Drawing(self.width, self.width, origin='center')
            
            if add_bg:
                #manually creating the white background
                canvas.append(draw.Rectangle(-self.width/2, -self.width/2, self.width, self.width, fill='white'))

            black = '#000000ff'

            for y in range(self.matrix_diamonds.shape[0]):
                for x in range(y, self.matrix_diamonds.shape[1]):
                    diamond = self.matrix_diamonds[y][x]
                    if add_line:
                        diamond.draw(canvas, stroke=black)
                    else:
                        diamond.draw(canvas)
                
            if plot_indicators:

                mid_size = int(self.width / 50.)
                eigth_size = int(mid_size /4.)
                quart_size = eigth_size * 3

                white_fill = '#ffffff00'
                middle_gray= '#666666ff'

                stroke_width = int(self.width / 1000) + 2

                #Middle points
                canvas.append(draw.Circle(-self.width/4, 0, mid_size,
                    fill=white_fill, fill_opacity=0, stroke='black', stroke_width = stroke_width))
                canvas.append(draw.Circle(self.width/4, 0, mid_size,
                    fill=white_fill, fill_opacity=0, stroke='black', stroke_width = stroke_width))

                #3 quarter points
                canvas.append(draw.Circle(-self.width/8, self.width/4, quart_size,
                                fill=white_fill, fill_opacity=0, stroke=middle_gray, stroke_width = stroke_width))
                canvas.append(draw.Circle(self.width/8, self.width/4, quart_size,
                    fill=white_fill, fill_opacity=0, stroke=middle_gray, stroke_width = stroke_width))

                #One quarter points
                canvas.append(draw.Circle(-3*self.width/8, -self.width/4, quart_size,
                                fill=white_fill, fill_opacity=0, stroke=middle_gray, stroke_width = stroke_width))
                canvas.append(draw.Circle(3*self.width/8, -self.width/4, quart_size,
                    fill=white_fill, fill_opacity=0, stroke=middle_gray, stroke_width = stroke_width))

                # all eight points, declared from top to bottom.                        
                canvas.append(draw.Circle(-self.width/16, 3*self.width/8, eigth_size,
                                fill=middle_gray, stroke=middle_gray))
                canvas.append(draw.Circle(self.width/16, 3*self.width/8, eigth_size,
                    fill=middle_gray, stroke=middle_gray))

                canvas.append(draw.Circle(-3*self.width/16, self.width/8, eigth_size,
                                fill=middle_gray, stroke=middle_gray))
                canvas.append(draw.Circle(3*self.width/16, self.width/8, eigth_size,
                                fill=middle_gray, stroke=middle_gray))

                canvas.append(draw.Circle(-5*self.width/16, -self.width/8, eigth_size,
                                fill=middle_gray, stroke=middle_gray))
                canvas.append(draw.Circle(5*self.width/16, -self.width/8, eigth_size,
                                fill=middle_gray, stroke=middle_gray))

                canvas.append(draw.Circle(-7*self.width/16, -3*self.width/8, eigth_size,
                                fill=middle_gray, stroke=middle_gray))
                canvas.append(draw.Circle(7*self.width/16, -3*self.width/8, eigth_size,
                    fill=middle_gray, stroke=middle_gray))
            return canvas

    def animate_draw(self, nmbr_seconds, frames_per_row=1):
        if self.matrix_diamonds is None or self.matrix_diamonds.shape[0] < 1 or self.matrix_diamonds.shape[1] < 1:
            raise Exception("cannot draw when there are nothing to draw. Don't forget to generate diamonds in the correct mode before drawing.")
        number_of_row = np.shape(self.matrix_diamonds)[0]
        time_per_row = float(nmbr_seconds)/float(number_of_row)
        time_for_one_iteration = time_per_row/float(frames_per_row)
        d = draw.Drawing(self.width, self.width, origin='center')

        d.append(draw.Rectangle(-self.width/2, -self.width/2, 0, 0, fill='white'))
        # Draw one frame of the animation
        def draw_frame(y_index ,opacity_as_hex, canvas):
            #need to iterate in this order for the drawing by columns.
            for x in range(y_index, self.matrix_diamonds.shape[1]):
                elem = self.matrix_diamonds[y_index][x]
                if elem:
                    elem.draw(canvas, elem.color+opacity_as_hex)
            return canvas
        
        with draw.animate_jupyter(draw_frame, delay=time_per_row/float(frames_per_row)) as anim:
            opacity_step = percentage_to_hex_opacity(1/float(frame_per_row))
            for y in range(number_of_row):
                for f in range(frames_per_row):
                    anim.draw_frame(y, opacity_step, d)


def add_label_top_left_corner_of_canvas(canvas, label, plot_size):
    text_size = 0.75*(plot_size/10) # we want the text to be displayed on one tenth of the plot's size
    x_spot = -2*(plot_size)/5.
    y_spot = -x_spot
    canvas.append(draw.Text(label, text_size, x_spot, y_spot, fill='black'))
    return canvas


def draw_point_on_the_circle(canvas, angle, label, color, arc_w, display_label=True):
    x = math.cos(angle)
    y = math.sin(angle)
    canvas.append(draw.Circle(x*arc_w, y*arc_w, 4,
        fill=color, stroke=color))
    if display_label:
        text_width = 20
        label_offset = 15
        x_label = x*(1+text_width/(2*arc_w))*(1+label_offset/arc_w)
        y_label = y*(1+text_width/(2*arc_w))*(1+label_offset/arc_w)
        canvas.append(draw.Text(label, text_width, x_label*arc_w, y_label*arc_w, fill=color))
    return canvas


def draw_circle_color_space(color_mapping_func, label_angle_dict, draw_width, add_zero_degree=True, add_labels=True):
    canvas_width = draw_width
    canvas_height = draw_width
    canvas = draw.Drawing(width=canvas_width, height=canvas_height, origin="center")
    #this does not want to work the way I want and I have no idea why...
    #canvas.append(draw.Rectangle(-canvas_width,-canvas_height,canvas_width,canvas_height, fill='white'))
    lower_arc_w = 0.2*draw_width
    higher_arc_w = 0.4*draw_width
    for i in range(-180, 180):
        angle_color = rgb_to_hex(color_mapping_func(i*(math.pi/180)))
        p = draw.Path(fill=angle_color)
        #i+2 so that the slight overlap hides the seams. 
        p.arc(0,0,higher_arc_w,i,i+2)
        p.arc(0,0,lower_arc_w,i+1,i,cw=True, includeL=True)
        canvas.append(p)
        
    for  label, angle in label_angle_dict.items():
        if label[-1] == 'm':
            draw_point_on_the_circle(canvas, angle, label, "#000000", lower_arc_w, add_labels)
        else:
            draw_point_on_the_circle(canvas, angle, label, "#777777", higher_arc_w, add_labels)
    if add_zero_degree:
        draw_point_on_the_circle(canvas, 0, '0°', '#000000', higher_arc_w)
    return canvas

#Kind of a weird function. Basically it will display all elements from `key_label_color_dict` as circles going from inside to outside,
# And on each of those circles, it will shift the elem_vector by the index of the circle and display elem vector on the circle.
def vizualise_all_transposition_circles(key_label_color_dict, elem_vector_color_dict, vector_to_angle_func, plot_width):
    d = draw.Drawing(plot_width, plot_width, origin='center')
    #background
    d.append(draw.Rectangle(-plot_width/2,-plot_width/2, plot_width, plot_width, fill='#EEEEEE'))
    
    n_keys = float(len(key_label_color_dict))
    
    margin = plot_width/n_keys
    paddin = 2*margin
    outside_border = plot_width-paddin
    step = (outside_border-paddin)/(2.*n_keys)
    idx = 0
    for _,v in key_label_color_dict.items():
        curr_circle_width = paddin+idx*step
        d.append(draw.Circle(0, 0, curr_circle_width, stroke_width=2, stroke=rgb_to_hex(v), fill_opacity=0.))
        idx += 1
        for k, t in elem_vector_color_dict.items():
            vector = np.roll(t[0], idx)
            angle = vector_to_angle_func(vector)
            color = rgb_to_hex(t[1])
            draw_point_on_the_circle(d, angle, '', color, curr_circle_width)
    
    return d


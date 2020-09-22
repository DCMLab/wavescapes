import numpy as np
import music21 as m21
import pretty_midi as pm
from madmom.audio import chroma

from warnings import warn
import tempfile
import math
import os

deep_chroma_processor = chroma.DeepChromaProcessor()

twelve_tones_vector_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#','A', 'A#', 'B']

# this is to correct the name found in the outputed notes from music21 parsing of symbolic data files.
# Since this library assumes enharmonic equivalence, any note's name should be mapped to one of the twelve values
# found in 'twelve_tones_vector_name'
altered_notation_dict = {
    'C-' : 'B',
    'C--': 'A#',
    'C---': 'A',
    'C##' : 'D',
    'C###' : 'D#',
    'D-' : 'C#',
    'D--' : 'C',
    'D---' : 'B',
    'D##' : 'E',
    'D###' : 'F',
    'E-' : 'D#',
    'E--' : 'D',
    'E---' : 'C#',
    'E#' : 'F',
    'E##' : 'F#',
    'E###' : 'G',
    'F-' : 'E',
    'F--' : 'D#',
    'F---' : 'D',
    'F##' : 'G',
    'F###' : 'G#',
    'G-' : 'F#',
    'G--' : 'F',
    'G---' : 'E#',
    'G##' : 'A',
    'G###' : 'A#',
    'A-' : 'G#',
    'A--' : 'G',
    'A---' : 'F#',
    'A##': 'B',
    'A###': 'C',
    'B-' : 'A#',
    'B--' : 'A',
    'B---' : 'G#',
    'B#' : 'C',
    'B##': 'C#',
    'B###': 'D',
} 

pitch_pitch_dict = {x: x for x in twelve_tones_vector_name}

# In the end we want each string to match an index between 0 and 11, so it fits inside a 12-d vector.
pitch_index_dict = {twelve_tones_vector_name[i]:i for i in range(len(twelve_tones_vector_name))}

# So any pitch name given to this dict will be mapped to its cannonical form defined in 'twelve_tones_vector_name'
normalize_notation_dict = dict(altered_notation_dict,  **pitch_pitch_dict)

def recursively_map_offset(stream, only_note_name=True):
    """
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
    stream : music21.stream.Stream
        the music21 stream to be parsed into the list of pitches and time offset.
        
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
        which starts at the 10th quarter note of the input file and stops one quarter note and half
        later in the piece. 
        
    
    """
    res = []
    for elem in stream.recurse():
        if isinstance(elem, m21.note.Note):
            start_offset = elem.getOffsetInHierarchy(stream)
            res.append((elem.name if only_note_name else elem, (start_offset, start_offset+elem.duration.quarterLength)))
        elif isinstance(elem, m21.chord.Chord):
            start_offset = elem.getOffsetInHierarchy(stream)
            res += list(map(lambda r: (r.name if only_note_name else r , (start_offset, start_offset+elem.duration.quarterLength)), elem.pitches))
    return res


def remove_unpitched_tracks_from_xml_stream(xml_stream):
    """
    Takes care of removing drum/percussions tracks from an xml file parsed into a .
    Work only if the XML file has metadata clearly indicating parts that have a 
    "percussion" clef (or no clef at all). 
    
    Parameters
    ----------
    xml_stream : music21.stream.Stream
        the music21 stream that needs to have percussive parts removed. 
    
    Returns
    -------
    instance of music21.stream.Stream without the percussive parts of the score
    
    """
    #creating a new score that will hold the pitched parts
    s = m21.stream.Score() 
    for part in m21.instrument.partitionByInstrument(xml_stream):
        #filtering out the non-pitched clefs.
        if not isinstance(part.clef, m21.clef.NoClef) and not isinstance(part.clef, m21.clef.PercussionClef):
            s.append(part)
    return s

def remove_unpitched_tracks_from_midi_file(midi_filepath):
    """
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
    
    """
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
    """
    the beat offset must be expressed as units of quarter notes. 
    Taken are all beat which at least END AFTER the beat1, and START BEFORE the beat2
    """
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
    """
    This functions transforms a list of tuples each containing the name of the pitch
    followed by its start and end in the file into a pitch class distribution with each
    pitch class given the weight corresponding to its duration in the current slice of
    temporal size aw_size.
    
    """
    max_beat = get_max_beat(pitch_offset_array)
    
    if aw_size <= max_beat/2:
        chunk_number = math.ceil(max_beat/aw_size)
    else:
        raise Exception('The analysis window\'s size (%lf) should not exceed half the duration of the musical piece (%lf).'%(aw_size, max_beat/2.))
    
    res_vector = np.full((chunk_number, 12), 0.0, np.float64)

    for b in range(chunk_number):
        start_beat = b*aw_size
        stop_beat = (b+1)*aw_size
        analysis_windows = slice_according_to_beat(pitch_offset_array, start_beat, stop_beat)
        pitch_class_vec = sum_into_pitch_class_vector(analysis_windows, start_beat, stop_beat)
        res_vector[b] = pitch_class_vec
    
    return res_vector

#useful to increase the default resolution 
def reframe_pc_mat(pc_mat, bin_size):
    if bin_size == 1:
        return pc_mat
    
    curr_bin_nb = pc_mat.shape[0]
    new_bin_nb = int(math.ceil(curr_bin_nb/bin_size))
    res = np.full((new_bin_nb, pc_mat.shape[1]), 0., np.float64)
    for i in range(new_bin_nb):
        res[i] = sum(pc_mat[int(i*bin_size):int((i+1)*bin_size)])
    return res

#left here for comparison purpose.
'''
def librosa_chromagram(filepath, aw_size):
    audio_array, sample_ratio = librosa.load(filepath)
    
    hop_len = round(sample_ratio*aw_size) #hop_len is the analysis window size for the chromagrams in terms of number of sample.
    #so the result's shape is consistent with the one produced in the case of midi files.
    return np.transpose(librosa.feature.chroma_stft(audio_array, sample_ratio, hop_length=hop_len))
'''

def madmom_chromagram(filepath, aw_size, deep_chroma = False):
    dcp_base_fps = 10
    fps = 1./float(aw_size)
    if aw_size < .1:
        raise Exception("Audio PCV extraction using the deep chroma extractor can not be done for resolution lower than 0.1 (one tenth of a second)")
    
    if deep_chroma:
        chromagrams = deep_chroma_processor.process(filepath)
        return reframe_pc_mat(chromagrams, (dcp_base_fps/fps)) 
    else:
        clp = chroma.CLPChromaProcessor(fps=fps)
        return clp.process(filepath)
    
# trim the input array so that no empty vectors are located at the beginning and end of the muscial piece
def trim_pcs_array(pcvs):
    start = 0
    while not np.any(pcvs[start]):
        start += 1
    end = len(pcvs) - 1
    while not np.any(pcvs[end]):
        end -= 1
    return pcvs[start:end+1]

def produce_pitch_class_matrix_from_filename(filepath, aw_size = 1., trim_extremities=True, remove_unpitched_tracks = False, deep_chroma = False):
    """
    This function takes a MIDI or XML file as a parameters and
    transforms it into "list of pitch class distribution"
    This list is modelised by a Nx12 matrix of float values, each 
    row of the matrix corresponds to the pitch content of one slice
    of temporal size "aw_size" from the musical content from the
    input's file. The number of row N depends on the temporal size
    of the file, and the number chosen for aw_size.
    
    Parameters
    ----------
    filepath : str 
        the path of the MID/MIDI/XML/MUSICXML/WAV file whose musical content 
        is transformed into the pitch class distribution's list.
        It should be noted that MIDI file exported by Musescore often features 
        "truncated" note's length for the playback to feel more natural, however this 
        affects the the Pitch Class Vectors produced from this function. For example,
        notes shorter than a sixteenth notes will simply be mapped to a weight of 0 in
        the resulting PCVs. If you use MuseScore to edit scores, always export your musical
        pieces in xml format, as this format does not suffer from those truncations. Only
        use MIDI if you are certain of the integrity of your file, or if you don't have any choices.
        Other audio file formats than WAV are supported, but they all rely on "ffmpeg" and as
        such are only supported if this package is installed.
        
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
        class distribution. Only apply to input data in symbolic format (i.e. not on real audio)
        Default value is True.
                 
    remove_unpitched_tracks : bool, optional 
        indicates whether percussive instruments need to be removed from the mix. 
        Only applies to file holding symbolic data (MIDI and XML currently).
        Default value is False.
        
    deep_chroma: bool, optional
        indicates which chromagram extractor to use from the madmom library.
        If set to True, the DeepChromaProcessor (a pretrained DNN with a set resolution
        of one tenth of a second) is used, otherwise, the CLPChromaProcessor 
        (a method of PCV retrival using log-compression with parametric resolution) 
        from madmom is used.
        Default value is False. 
    
    Returns
    -------
    numpy matrix of shape Nx12 (numpy.ndarray of numpy.float64)
        This matrix holds the pitch distributions corresponding to all
        the pitch content from all non overlapping slices of aw_size size from the file
        given as argument.
    
    """
    lower_filepath = filepath.lower()
    midi_extensions = ('.mid', '.midi')
    xml_extensions = ('.mxl', '.xml', '.musicxml')

    if not os.path.isfile(filepath):
        raise Exception('Cannot find file "%s"'%filepath)
    
    if lower_filepath.endswith(midi_extensions) or lower_filepath.endswith(xml_extensions):

        if deep_chroma:
            msg = "argument 'deep_chroma' is meaningless on Symbolic data input. Only use it for real audio data."
            warn(msg)
        
        music_stream = None
        if remove_unpitched_tracks:
            if lower_filepath.endswith(midi_extensions):
                music_stream = m21.converter.parse(remove_unpitched_tracks_from_midi_file(filepath))
            elif lower_filepath.endswith(xml_extensions):
                music_stream = remove_unpitched_tracks_from_xml_stream(m21.converter.parse(filepath))
        else:
            #someone wants to keep non-pitched elements in their wavescape. K.
            music_stream = m21.converter.parse(filepath)
        
        pitch_offset_list = recursively_map_offset(music_stream)
        pcvs_arr = pitch_class_set_vector_from_pitch_offset_list(pitch_offset_list, aw_size)
    else:
        # audio file input.
        if remove_unpitched_tracks:
            msg = "'remove_unpitched_tracks' argument is meaningless on real audio. Only use it for symbolic data (MIDI/XML)."
            warn(msg)
        pcvs_arr = madmom_chromagram(filepath, aw_size, deep_chroma)
        
    return trim_pcs_array(pcvs_arr) if trim_extremities else pcvs_arr

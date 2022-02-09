import pytest
import os, math
from itertools import product
import numpy as np

from wavescapes.pcv import produce_pitch_class_matrix_from_filename
from wavescapes.dft import apply_dft_to_pitch_class_matrix, clip_normalized_floats, normalize_dft
from wavescapes.color import complex_utm_to_ws_utm

from warnings import warn


def rgba_to_rgb(to_convert, background=None):
    """Takes an RGBA tuple of 4 floats between [0,1] and eliminates the alpha channel by mixing
    R, G, and B with the background colour (which defaults to white).
    """
    if background is None:
        background = (1., 1., 1.,)
    if len(to_convert) == 3:
        return to_convert  # no point converting something that is already in RGB
    if len(to_convert) != 4:
        raise Exception('Incorrect format for the value to be converted, should have length of 4')
    if len(background) != 3:
        raise Exception('Incorrect format for the value background, should have length of 3 '
                        '(no alpha channel for this one)')
    alpha = to_convert[3]
    return tuple([alpha * convert_c + (1 - alpha) * background_c for convert_c, background_c in
                  zip(to_convert, background)])


def stand(v):
    if v > 1:
        if np.isclose(v, 1):
            return 255
        assert v <= 1, f"Expected RGBA channel to be <= 1 but got {v}"
    return int(v * 0xff)


def previous_complex_utm_to_ws_utm(utm, coeff, magn_stra='0c', output_rgba=False, output_raw_values=False,
                  ignore_magnitude=False, ignore_phase=False, decimal=False):
    """
    Converts an upper triangle matrix filled with Fourier coefficients into
    an upper triangle matrix filled with color values that serves as the mathematical model
    holding the color information needed to build the wavescapes plot.

    Parameters
    ----------
    utm : numpy matrix shape NxN (numpy.ndarray of numpy.complex128)
        An upper triangle matrix holding all fourier coefficients 0 to 6 at all different hierarchical levels

    coeff: int
        number between 1 to 6, will define which coefficient plot will be visualised in the outputted upper triangle matrix.

    magn_stra : {'0c', 'boost', 'max', 'max_weighted', 'raw'}, optional
        Since the magnitude is unbounded, but its grayscale visual representation needs to be bounded,
        Different normalisation of the magnitude are possible to constrain it to a value between 0 and 1.
        Below is the listing of the different value accepted for the argument magn_stra
        - '0c' : default normalisation, will normalise each magnitude by the 0th coefficient
            (which corresponds to the sum of the weight of each pitch class). This ensures only
            pitch class distribution whose periodicity exactly match the coefficient's periodicity can
            reach the value of 1.
        - 'post_norm' : based on the 0c normalisation but "boost" the space of all normalized magnitude so
                    the maximum magnitude observable is set to the max opacity value. This means that if any PCV in the
                    utm given as input reaches the 0c normalized magnitude of 1, this parameter acts like
                    the '0c' one. This magn_strat should be used with audio input mainly, as seldom PCV derived
                    from audio data can reach the maximal value of normalized magnitude for any coefficient.
        - 'max': set the grayscal value 1 to the maximum possible magnitude in the wavescape, and interpolate linearly
            all other values of magnitude based on that maximum value set to 1. Warning: will bias the visual representation
            in a way that the top of the visualisation will display much more magnitude than lower levels.
        - 'max_weighted': same principle as max, except the maximum magnitude is now taken at the hierarchical level,
            in other words, each level will have a different opacity mapping, with the value 1 set to the maximum magnitude
            at this level. This normalisation is an attempt to remove the bias toward higher hierarchical level that is introduced
            by the 'max' magnitude process cited previously.
        - 'raw' : does not normalize the magnitude at all. Can break the wavescapes pipeline as the
            raw magnitude values cannot be mapped in
        Default value is '0c'

    ignore_magnitude : bool, optional
        Determines whether to remove the opacity mapping from the color mapping
        Default value is False (meaning opacity mapping is enabled by default)

    ignore_phase : bool, optional
        Determines whether to remove the hue mapping from the color mapping
        Default value is False (meaning hue mapping is enabled by default)

    output_rgba : bool, optional
        Determines whether the resulting matrix contains color values encoded in RGB (Red Green Blue) or
        RGBA (Red Green Blue Alpha) format, where alpha is the opacity channel.
        RGBA can be useful in opacity mapping to obtain wavescapes that are somewhat transparent on low magnitude elements.
        If output_rgba is set to false (i.e. values outputed in RGB instead), the opacity is performed by assuming a white backdrop, and
        generating the rgb values of the transparent color over white.
        This argument has a side-effet later in the pipeline during Wavescape.draw. Because of the rasterisation process
        of Matplotlib, transparent elements of the wavescapes will have noticeable seams between them.
        The higher the pixel size of the plots relative to the number of elements will make those seams
        appear less noticeable, but so far no concrete solution exists to remove those seams. Those seams
        are not present in RGB mode.
        Note that you might need the "transparent" parameters of  the function plt.savefig to be set to True in order
        to correctly save the plot as a transparent image.
        Note that this argument works with saturation, but nothing changes (the alpha channel
        is just set to 1 in RGBA mode anyway) except the seams as a result of the RGBA rasterisation side effet.
        Default value is False (rgb values are outputted)

    output_raw_values: bool, optional
        Determines whether the resulting matrix contains tuples of raw angle and magnitude (normalized or not)
        that were not converted to color values. This can be useful for using wavescapes in other purposes than
        visualisation (for instance, using them to do corpus analysis or historical analysis). Since this type
        of output is incompatible with visualisation without any further processing, this parameter is not
        accessible through the aggregate function "generate_single_wavescape" and "generate_all_wavescapes"
        Default value is False (rgb(a) values are outputted)

    Returns
    -------
    numpy.ndarray
        an upper triangle matrix of dimension NxNx2, NxNx3 or NxNx4 holding either rgb(a)
        values corresponding to the color mapping of a single Fourier coefficient from the input, or
        the angle and magnitudes of a certain coefficient.
    """

    def zeroth_coeff_cm(value, coeff):
        zero_c = value[0].real
        if zero_c == 0.:
            # empty pitch class vector, thus returns white color value.
            # this avoid a nasty divide by 0 error two lines later.
            return (0., 0.)  # ([0xff]*3
        nth_c = value[coeff]
        magn = np.abs(nth_c) / zero_c
        angle = np.angle(nth_c)
        return (magn, angle)

    def max_cm(value, coeff, max_magn):
        if max_magn == 0.:
            return (0., 0.)
        nth_c = value[coeff]
        magn = np.abs(nth_c)
        angle = np.angle(nth_c)
        return (magn / max_magn, angle)

    if output_rgba and output_raw_values:
        output_rgba = False  # Only one shall prevail
        msg = "parameters 'output_rgba' and 'output_raw_values' cannot be set to True at the same time. " \
              "'output_raw_values' takes precedence in that case, and no color values are produced."
        warn(msg)

    shape_x, shape_y = np.shape(utm)[:2]
    # RGB => 3 values, RGBA => RGB + 1 value, raw values => angle & magnitude => 2 values
    channel_nbr = 4 if output_rgba else 2 if output_raw_values else 3
    default_value = 0.0 if output_raw_values or decimal else (0xff + 1)
    default_type = np.float64 if output_raw_values or decimal else np.uint64
    # +1 to differentiate empty elements from white elements later down the line.
    res = np.full((shape_x, shape_y, channel_nbr), default_value, default_type)

    if magn_stra == '0c':
        for grouped_segments in range(shape_x):
            for last_segment in range(shape_y):
                curr_value = utm[grouped_segments][last_segment]
                if np.any(curr_value):  # this filters out the lower triangle but also empty segments
                    magn, angle = zeroth_coeff_cm(curr_value, coeff)
                    res[grouped_segments][last_segment] = previous_circular_hue(angle, magnitude=magn,
                                                                       output_rgba=output_rgba,
                                                                       ignore_magnitude=ignore_magnitude,
                                                                       ignore_phase=ignore_phase,
                                                                       decimal=decimal) if not output_raw_values else (
                    magn, angle)

    elif magn_stra == 'post_norm':
        magn_angle_mat = np.full((shape_x, shape_y, 2), 0., np.float64)
        for grouped_segments in range(shape_x):
            for last_segment in range(shape_y):
                curr_value = utm[grouped_segments][last_segment]
                if np.any(curr_value):  # this filters out the lower triangle but also empty segments
                    magn, angle = zeroth_coeff_cm(curr_value, coeff)
                    magn_angle_mat[grouped_segments][last_segment][0] = magn
                    magn_angle_mat[grouped_segments][last_segment][1] = angle
        max_magn = np.max(magn_angle_mat[:, :, 0])
        boosting_factor = 1. / float(max_magn)
        msg = 'Max magnitude of %lf observed for coeff. number %d, post normalizing all magnitudes by %.2lf%% of their ' \
              'original values' % (max_magn, coeff, 100 * boosting_factor)
        print(msg)

        for grouped_segments in range(shape_x):
            for last_segment in range(shape_y):
                magn, angle = magn_angle_mat[grouped_segments][last_segment]
                if np.any([magn, angle]):  # this filters out the lower triangle but also empty segments
                    res[grouped_segments][last_segment] = previous_circular_hue(angle,
                                                                       magnitude=magn * boosting_factor,
                                                                       output_rgba=output_rgba,
                                                                       ignore_magnitude=ignore_magnitude,
                                                                       ignore_phase=ignore_phase,
                                                                       decimal=decimal) if not output_raw_values else (
                    magn * boosting_factor, angle)

    elif magn_stra == 'max':
        # arr[:,:,coeff] is a way to select only one coefficient from the tensor of all 6 coefficients
        max_magn = np.max(np.abs(utm[:, :, coeff]))
        for grouped_segments in range(shape_x):
            for last_segment in range(shape_y):
                curr_value = utm[grouped_segments][last_segment]
                if np.any(curr_value): # this filters out the lower triangle but also empty segments
                    magn, angle = max_cm(curr_value, coeff, max_magn)
                    res[grouped_segments][last_segment] = previous_circular_hue(angle, magnitude=magn,
                                                                       output_rgba=output_rgba,
                                                                       ignore_magnitude=ignore_magnitude,
                                                                       ignore_phase=ignore_phase,
                                                                       decimal=decimal) if not output_raw_values else (
                    magn, angle)

    elif magn_stra == 'max_weighted':
        for grouped_segments in range(shape_x):
            line = utm[grouped_segments]
            max_magn = np.max([np.abs(el[coeff]) for el in line])
            for last_segment in range(shape_y):
                curr_value = utm[grouped_segments][last_segment]
                if np.any(curr_value):  # this filters out the lower triangle but also empty segments
                    magn, angle = max_cm(curr_value, coeff, max_magn)
                    res[grouped_segments][last_segment] = previous_circular_hue(angle, magnitude=magn,
                                                                       output_rgba=output_rgba,
                                                                       ignore_magnitude=ignore_magnitude,
                                                                       ignore_phase=ignore_phase,
                                                                       decimal=decimal) if not output_raw_values else (
                    magn, angle)

    elif magn_stra == 'raw':
        for grouped_segments in range(shape_x):
            for last_segment in range(shape_y):
                curr_value = utm[grouped_segments][last_segment]
                if np.any(curr_value):  # this filters out the lower triangle but also empty segments
                    value = curr_value[coeff]
                    angle = np.angle(value)
                    magn = np.abs(value)
                    res[grouped_segments][last_segment] = previous_circular_hue(angle, magnitude=magn,
                                                                       output_rgba=output_rgba,
                                                                       ignore_magnitude=ignore_magnitude,
                                                                       ignore_phase=ignore_phase,
                                                                       decimal=decimal) if not output_raw_values else (
                    magn, angle)
    else:
        raise Exception('Unknown option for magn_stra')

    return res





def previous_circular_hue(angle, magnitude=1., output_rgba=True, ignore_magnitude=False, ignore_phase=False,
                 decimal=False):
    # np.angle returns value in the range of [-pi : pi], where the circular hue is defined for
    # values in range [0 : 2pi]. Rather than shifting by a pi, the solution is for the negative
    # part to be mapped to the [pi: 2pi] range which can be achieved by a modulo operation.
    def two_pi_modulo(value):
        return np.mod(value, 2 * math.pi)

    def step_function_quarter_pi_activation(lo_bound, hi_bound, value):
        # in the increasing path branch
        if value >= lo_bound and value <= lo_bound + math.pi / 3:
            return ((value - lo_bound) / (math.pi / 3))
        # in the decreasing path branch
        elif value >= hi_bound and value <= hi_bound + math.pi / 3:
            return 1 - ((value - hi_bound) / (math.pi / 3))
        else:
            # the case of red
            if lo_bound > hi_bound:
                return 0 if value > hi_bound and value < lo_bound else 1
            else:
                return 1 if value > lo_bound and value < hi_bound else 0

    # Need to shift the value with one pi as the range of the angle given is between pi and minus pi
    # and the formulat I use goes from 0 to 2pi.
    angle = two_pi_modulo(angle)
    green = lambda a: step_function_quarter_pi_activation(0, math.pi, a)
    blue = lambda a: step_function_quarter_pi_activation(math.pi * 2 / 3, math.pi * 5 / 3, a)
    red = lambda a: step_function_quarter_pi_activation(math.pi * 4 / 3, math.pi / 3, a)
    gray = lambda v: 1 - v
    if ignore_magnitude and not ignore_phase:
        value = (red(angle), green(angle), blue(angle))
        if output_rgba:
            value = value + (1.,)
    elif ignore_phase and not ignore_magnitude:
        g = gray(magnitude)
        value = (g, g, g, g) if output_rgba else (g, g, g)
    elif ignore_phase and ignore_magnitude:
        value = (0., 0., 0., 0.,) if output_rgba else (0., 0., 0.)
    else:
        value = (red(angle), green(angle), blue(angle), magnitude)
        if not output_rgba:
            # gets rid of alpha by mixing the background color into the RGB channels
            value = rgba_to_rgb(value, background=None)
    if not decimal:
        return tuple([stand(ch) for ch in value])
    return value


########################### TESTING PART



NORM_METHODS = ('raw', 'max_weighted', 'max', '0c', 'post_norm')

@pytest.fixture(
    params=['NCandP.mid',
            'trim_extremities_test.mid',
            'emptyspace.mid',
            #'AveMaria_desPrezmid.mid', music21 fails to parse this
            'palestrina.mid',
            'palestrina2.mid',
            '210606-Prelude_No._1_BWV_846_in_C_Major.mid','chopin-prelude-op28-2.mid',
            'Faust_Symphony_mov1.mid',
            'Op74_No2_Pure.mid',
            'giant_steps_first_chorus.mid',
            '38088_Ligeti-Etude-No-2-Cordes-a-vide.mid'],
    ids=['percussion',
         'beginning_gap',
         'middle_gap',
         #'desprez',
         'palestrina1',
         'palestrina2',
         'bach',
         'chopin',
         'liszt',
         'scriabin',
         'coltrane',
         'ligeti'])
def dft(request):
    """Providing one DFT upper triangle matrix after another to the test methods below."""
    rel_midi_path = os.path.join("../midiFiles", request.param)
    abs_midi_path = os.path.abspath(rel_midi_path)
    pc_mat = produce_pitch_class_matrix_from_filename(abs_midi_path, aw_size=4.)
    return apply_dft_to_pitch_class_matrix(pc_mat)

class TestColor:

    @pytest.mark.parametrize(["how", "coeff"],
                             product(NORM_METHODS, range(1,7)))
    def test_normalization(self, dft, how, coeff):
        aa = previous_complex_utm_to_ws_utm(dft, coeff, magn_stra=how, output_raw_values=True)
        bb = normalize_dft(dft, how=how, coeff=coeff)
        normalize_dft(dft, how=how, coeff=coeff)
        assert np.allclose(aa, bb), f"{how} normalization, coeff={coeff}"
        print(f"{how} normalization, coeff={coeff}: OK")

    @pytest.mark.parametrize(["how", "indulge_prototypes"],
                             product(NORM_METHODS, (False, True)))
    def test_batch_normalization(self, dft, how, indulge_prototypes):
        batch_normalized = normalize_dft(dft, how=how, coeff=None, indulge_prototypes=indulge_prototypes)
        if how != 'raw':
            mags, msg = clip_normalized_floats(batch_normalized[...,0])
            assert len(msg) == 0, f"'{how}' ({indulge_prototypes}) normalization failed: " + msg
        individuals = [normalize_dft(dft, how=how, coeff=i, indulge_prototypes=indulge_prototypes)
                       for i in range(1, 7)]
        composed_normalized = np.stack(individuals, axis=2)
        assert np.allclose(composed_normalized, batch_normalized), f"({indulge_prototypes}) Normalizing all 6 by {how} " \
            f"gives other result than normalizing them individually."
        print(f"({indulge_prototypes}) Normalizing all 6 by {how}: OK")

    @pytest.mark.parametrize(["coeff", "how", "rgba", "ig_mag", "ig_phase"],
                             product(range(1, 7),
                                     (n for n in NORM_METHODS if n != 'raw'),
                                     (False, True),
                                     (False, True),
                                     (False, True),))
    def test_decimal_rgb(self, dft, coeff, how, rgba, ig_mag, ig_phase):
        a = previous_complex_utm_to_ws_utm(dft, coeff, magn_stra=how, output_rgba=rgba, ignore_magnitude=ig_mag, ignore_phase=ig_phase, decimal=True)
        b =          complex_utm_to_ws_utm(dft, coeff, magn_stra=how, output_rgba=rgba, ignore_magnitude=ig_mag, ignore_phase=ig_phase, as_html=False)
        info_str = f"normalization: {how}, rgba={rgba}, coeff={coeff}, ignore_magnitude={ig_mag}, ignore_phase={ig_phase}: "
        if not np.allclose(a, b):
            mask = ~np.isclose(a, b)
            a_vals, b_vals = a[mask], b[mask]
            if not np.isclose((a_vals + b_vals).sum(), mask.sum()): # this is to allow for the different behaviours regarding rests:
                                                                    # previously, they were left at 0 (black) but natur
                assert False, info_str + f"Differences: {np.concatenate((a[np.newaxis, mask], b[np.newaxis, mask])).T}"
        print(info_str + 'OK')




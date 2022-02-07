import os, math
import numpy as np
from matplotlib.colors import hsv_to_rgb, to_hex
from pcv import produce_pitch_class_matrix_from_filename
from dft import apply_dft_to_pitch_class_matrix, reset_tril
from color import complex_utm_to_ws_utm


def complex2color(utm, coeff, magn_stra='0c', output_rgba=False, output_raw_values=False,
                  ignore_magnitude=False, ignore_phase=False):
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
    mag_phase_mx = normalize_dft(utm, how=magn_stra, coeff=coeff)
    if output_raw_values:
        return mag_phase_mx
    elif magn_stra == 'raw':
        print("Raw (absolute) magnitudes cannot be converted to colors. Returning the raw "
              "magnitude-phase matrix instead.")
        return mag_phase_mx
    return phases2color(mag_phase_mx, output_rgba, ignore_magnitude, ignore_phase)





def phases2color(mag_phase_mx, output_rgba=False, ignore_magnitude=False, ignore_phase=False, deg=False, background=None,
                 as_html=False):
    # np.angle returns value in the range of [-pi : pi], where the circular hue is defined for
    # values in range [0 : 2pi]. Rather than shifting by a pi, the solution is for the negative
    # part to be mapped to the [pi: 2pi] range which can be achieved by a modulo operation.

    n = mag_phase_mx.shape[0]
    if not (ignore_magnitude and ignore_magnitude):
        mags = mag_phase_mx[..., 0]
        assert mags.max() <= 1, "Magnitudes are expected to be <= 1"
    if not ignore_phase:
        angles = mag_phase_mx[..., 1] % math.tau
        if deg:
            angles = angles / 360.
        else:
            angles = angles / (2 * np.pi)
    if ignore_phase:
        dim = 4 if output_rgba else 3
        if ignore_magnitude:
            if as_html:
                nothing = "#00000000" if output_rgba else "#000000"
                return np.full((n, n), nothing)
            else:
                return np.zeros((n,n,dim))
        rgb = np.dstack([1-mags] * dim)
        reset_tril(rgb)
        if not as_html:
            return rgb
        return np.apply_along_axis(to_hex, 2, rgb, output_rgba)
    # interpret normalized phases as hue and convert to HSV colors by adding S=V=1
    sv_dimensions = np.ones((n, n, 2))
    hsv = np.dstack((angles, sv_dimensions))
    reset_tril(hsv)
    rgb = hsv_to_rgb(hsv)
    if output_rgba:
        if ignore_magnitude:
            rgb = np.dstack((rgb, np.ones((n,n,1))))
        else:
            rgb = np.dstack((rgb, mags))
    elif not ignore_magnitude:
        if background is None:
            bckg = (1 - mags)[...,None]
        else:
            bckg = (1 - mags)[..., None] * np.broadcast_to(background, (n,n,3))
        rgb = mags[...,None] * rgb + bckg
    reset_tril(rgb)
    if not as_html:
        return rgb
    return np.apply_along_axis(to_hex, 2, rgb, output_rgba)


def normalize_dft(dft=None, how='0c', coeff=None):
    if coeff is None:
        mags, phases = np.abs(dft[..., 1:]), np.angle(dft[..., 1:])
    else:
        mags, phases = np.abs(dft[..., coeff]), np.angle(dft[..., coeff])

    def concat_mags_phases():
        """Produce the function result by concatenating magnitudes and phases."""
        if coeff is None:
            return np.stack((mags, phases), axis=-1)
        return np.dstack((mags, phases))

    if how == 'raw':
        return concat_mags_phases()

    if how in ('0c', 'post_norm'):
        if coeff is None:
            norm_by = np.real(dft[..., [0]])
        else:
            norm_by = np.real(dft[..., 0])
    elif how == 'max':
        if coeff is None:
            norm_by = mags.max(axis=1, keepdims=True).max(axis=0, keepdims=True)
        else:
            norm_by = mags.max()
    elif how == 'max_weighted':
        if coeff is None:
            norm_by = mags.max(axis=-2, keepdims=True)
        else:
            norm_by = mags.max(axis=-1, keepdims=True)
    mags = np.divide(mags, norm_by, out=np.empty_like(mags), where=norm_by > 0)
    reset_tril(mags)
    if (mags > 1).any():
        print("After normalizing by coefficient 0, there are still values > 1. This can happen "
              "only where c0 is 0 or if a coefficient's magnitude is bigger than c0's.")
    if how == 'post_norm':
        if coeff is None:
            mags = mags / mags.max(axis=1, keepdims=True).max(axis=0, keepdims=True)
        else:
            mags = mags / mags.max()
    return concat_mags_phases()


NORM_METHODS = ('raw', 'max_weighted', 'max', '0c', 'post_norm')


def test_normalization(dft):
    for how in NORM_METHODS:
        for i in range(1,7):
            aa = complex_utm_to_ws_utm(dft, i, magn_stra=how, output_raw_values=True)
            bb = normalize_dft(dft, how=how, coeff=i)
            assert np.allclose(aa, bb), f"{how} normalization, coeff={i}"
            print(f"{how} normalization, coeff={i}: OK")

def test_batch_normalization(dft):
    for how in (how for how in NORM_METHODS if how != 'raw'):
        batch_normalized = normalize_dft(dft, how=how)
        assert (batch_normalized[..., 0] <= 1).all(), f"Normalizing all 6 by {how}: Values > 1 remained!"
        t = [normalize_dft(dft, how=how, coeff=i) for i in range(1, 7)]
        composed_normalized = np.stack([normalize_dft(dft, how=how, coeff=i) for i in range(1,7)], axis=2)
        z = complex_utm_to_ws_utm(dft, 1, magn_stra=how, output_raw_values=True)
        assert np.allclose(composed_normalized, batch_normalized), f"Normalizing all 6 by {how} " \
            f"gives other result than normalizing them individually."
        print(f"Normalizing all 6 by {how}: OK")

def test_decimal_rgb(dft):
    for how in (how for how in NORM_METHODS if how != 'raw'):
        for rgba in (False, True):
            for im in (False, True):
                for ip in (False, True):
                    for i in range(1,7):
                        a = complex_utm_to_ws_utm(dft, i, magn_stra=how, output_rgba=rgba, ignore_magnitude=im, ignore_phase=ip, decimal=True)
                        b = complex2color(dft, i, magn_stra=how, output_rgba=rgba, ignore_magnitude=im, ignore_phase=ip)
                        info_str = f"normalization: {how}, rgba={rgba}, coeff={i}, ignore_magnitude={im}, ignore_phase={ip}: "
                        assert np.allclose(a, b), info_str + 'FAIL'
                        print(info_str + 'OK')


bach_prelude_midi = os.path.abspath("../midiFiles/210606-Prelude_No._1_BWV_846_in_C_Major.mid")
pc_mat_bach = produce_pitch_class_matrix_from_filename(bach_prelude_midi, aw_size=4.)
dft_mat_bach = apply_dft_to_pitch_class_matrix(pc_mat_bach)
test_decimal_rgb(dft_mat_bach)
test_batch_normalization(dft_mat_bach)
test_normalization(dft_mat_bach)
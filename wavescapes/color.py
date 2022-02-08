import numpy as np
import math

from matplotlib.colors import to_hex, hsv_to_rgb

from .dft import reset_tril

def clip_normalized_floats(arr):
    """Checks for values below 0 and above 1, warns, and returns a clipped copy.

    Parameters
    ----------
    arr : np.array
        Array of any dimension containing floats.

    Returns
    -------
    np.array, str
        Returns a clipped copy and a short report.
    """
    surpassing = arr > 1.
    subpassing = arr < 0.
    above, below = surpassing.any(), subpassing.any()
    if above or below:
        fails = surpassing.sum() + subpassing.sum()
        plural = fails > 1
        msg = 'There were ' if plural else 'There was '
        if below:
            n_below = subpassing.sum()
            mean_below = np.mean(arr[subpassing])
            msg += f"{n_below} value"
            msg += f"s < 0 (mean={mean_below}) " if n_below > 1 else f" < 0 ({mean_below}) "
        if above:
            n_above = surpassing.sum()
            mean_above = np.mean(arr[surpassing] - 1.)
            if below:
                msg += 'and '
            msg += f"{n_above} value"
            msg += f"s > 1 (mean={mean_above}) " if n_above > 1 else f" > 1 ({mean_above}) "
        msg += "which had to be clipped."
        return np.clip(arr, 0, 1), msg
    return arr, ''


def atleast_2d(arr):
    """Variant of np.atleast_2d that makes arr last not first."""
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    elif arr.ndim == 1:
        return arr[:, np.newaxis]
    else:
        return arr


def circular_hue(mag_phase_mx, output_rgba=False, ignore_magnitude=False, ignore_phase=False,
                 deg=False, background=None,
                 as_html=True):
    """ Transforms a matrix of magnitudes and phases into HTML or normalized RGB(A) colors.

    Parameters
    ----------
    mag_phase_mx : np.array
        (..., 2) array where the last dimension has [mag, phase] pairs. Magnitudes need to be
        normalized.
    output_rgba : bool, optional
        If True, add an alpha channel to the color.
    ignore_magnitude : bool, optional
        All colors get full opacity without taking the magnitude into account.
    ignore_phase : bool, optional
        Show only magnitudes on the greyscale without colours.
    deg : bool, optional
        Pass True if the angles are expressed as degrees rather than radians.
    background : collection, optional
        In the case that the background colour should be other than white and output_rgba is False,
        pass the background colour as normalized RGB tuple (i.e. three values between [0,1]).
    as_html : bool, optional
        Defaults to True, meaning that the colors are returned as strings of length 7 (RGB) or
        9 (RGBA). Pass False to get normalized RGB arrays (i.e. three or four values between [0,1],
        depending on output_rgba) on the last dimension.

    Returns
    -------
    nd.array
        Same shape except for the last dimension which will have size 1 if as_html is True (default),
        3 if False, and 4 if False and output_rgba is True.
    """
    mag_phase_mx = atleast_2d(mag_phase_mx)
    *dims, lastdim = mag_phase_mx.shape
    assert lastdim == 2, f"Expecting a magnitude-phase matrix, i.e. the last dimension needs to be 2, not {lastdim}."
    other_dimensions = tuple(dims)
    if len(other_dimensions) == 1:
        is_square = False
    else:
        one, two, *_ = other_dimensions
        is_square = one == two
    if not (ignore_magnitude and ignore_magnitude):
        mags = mag_phase_mx[..., [0]]
        mags, msg = clip_normalized_floats(mags)
        if len(msg) > 0:
            msg = "Incorrect magnitudes passed to circular_hue(): " + msg
            print(msg)
    if not ignore_phase:
        angles = mag_phase_mx[..., [1]] % math.tau
        if deg:
            angles = angles / 360.
        else:
            angles = angles / math.tau
    if ignore_phase:
        dims = 4 if output_rgba else 3
        if ignore_magnitude:
            if as_html:
                nothing = "#00000000" if output_rgba else "#000000"
                return np.full(other_dimensions, nothing)
            else:
                return np.zeros(other_dimensions + (dims,))
        rgb = np.dstack([1 - mags] * dims)
        if is_square:
            reset_tril(rgb)
        if not as_html:
            return rgb
        return np.apply_along_axis(to_hex, 2, rgb, output_rgba)
    # interpret normalized phases as hue and convert to HSV colors by adding S=V=1
    sv_dimensions = np.ones(other_dimensions + (2,))
    hsv = np.concatenate((angles, sv_dimensions), axis=-1)
    if is_square:
        reset_tril(hsv)
    rgb = hsv_to_rgb(hsv)
    if output_rgba:
        if ignore_magnitude:
            rgb = np.concatenate((rgb, np.ones(other_dimensions + (1,))), axis=-1)
        else:
            rgb = np.concatenate((rgb, mags), axis=-1)
    elif not ignore_magnitude:
        # this is the default case where no alpha channel is returned so that the opacity that
        # reflects the magnitude is achieved by merging the colors with the background color
        if background is None:
            rgb_dims = rgb.shape
            bckg = np.broadcast_to(1. - mags, rgb_dims) # the amount of white
        else:
            bckg = np.broadcast_to(1. - mags, rgb_dims) * np.broadcast_to(background, rgb_dims)
        rgb = np.broadcast_to(mags, rgb_dims) * rgb + bckg
    if is_square:
        reset_tril(rgb)
    if not as_html:
        return rgb
    return np.apply_along_axis(to_hex, -1, rgb, output_rgba)


def complex_utm_to_ws_utm(utm, coeff, magn_stra='0c', output_rgba=False, output_raw_values=False,
                          ignore_magnitude=False, ignore_phase=False, as_html=True):
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
    return circular_hue(mag_phase_mx, output_rgba, ignore_magnitude, ignore_phase, as_html=as_html)


def normalize_dft(dft=None, how='0c', coeff=None):
    """ Converts complex numbers into magnitude and phase and normalizes the magnitude by one of
    several possible approaches.

    Parameters
    ----------
    dft : np.array
        An (NxNx7) upper triangular matrix of complex numbers (although the last dimension could
        be larger in principle).
    how : {'0c', 'post_norm', 'max', 'max_weighted', 'raw'}
        See above.
    coeff : int, optional
        By default, the normalization is performed on all coefficients and an (NxNx7x2) matrix is
        returned. Pass an integer to select only one of them.

    Returns
    -------
    np.array
        (NxNx7x2) if coeff is None, else (NxNx2). In both cases the last dimension contains
        (normalized magnitude, phase) pairs.
    """
    if coeff is None:
        mags, phases = np.abs(dft[..., 1:]), np.angle(dft[..., 1:])
    else:
        mags, phases = np.abs(dft[..., coeff]), np.angle(dft[..., coeff])

    def concat_mags_phases():
        """Produce the function result by concatenating magnitudes and phases."""
        nonlocal mags, phases
        if not 'raw':
            mags, msg = clip_normalized_floats(mags)
            if len(msg) > 0:
                msg = f"Normalizing by '{how}' left unnormalized values: " + msg
                print(msg)
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
    if how == 'post_norm':
        if coeff is None:
            mags = mags / mags.max(axis=1, keepdims=True).max(axis=0, keepdims=True)
        else:
            mags = mags / mags.max()
    return concat_mags_phases()
import numpy as np
import math

from matplotlib.colors import to_hex, hsv_to_rgb

from .dft import reset_tril, clip_normalized_floats, normalize_dft


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
    mags = mag_phase_mx[..., [0]]
    mags, msg = clip_normalized_floats(mags)
    if len(msg) > 0:
        msg = "Incorrect magnitudes passed to circular_hue(), use normalize_dft() before. " + msg
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
                nothing = "#ffffffff" if output_rgba else "#ffffff"
                return np.full(other_dimensions, nothing)
            else:
                return np.ones(other_dimensions + (dims,))
        rgb = np.dstack([1 - mags] * dims)
        if is_square:
            reset_tril(rgb)
        if not as_html:
            return rgb
        return np.apply_along_axis(to_hex, -1, rgb, output_rgba)
    # interpret normalized phases as hue and convert to HSV colors by adding S=V=1
    s_and_v_dimension = mags.any(axis=-1, keepdims=True) * 1.
    hsv = np.concatenate((angles, s_and_v_dimension, s_and_v_dimension), axis=-1)
    rgb = hsv_to_rgb(hsv)
    if output_rgba:
        if ignore_magnitude:
            rgb = np.concatenate((rgb, np.ones(other_dimensions + (1,))), axis=-1)
        else:
            rgb = np.concatenate((rgb, mags), axis=-1)
    elif not ignore_magnitude:
        # this is the default case where no alpha channel is returned so that the opacity that
        # reflects the magnitude is achieved by merging the colors with the background color
        rgb_dims = rgb.shape
        if background is None:
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
                          ignore_magnitude=False, ignore_phase=False, as_html=True,
                          indulge_prototype=False):
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
        Normalization method used by :py:func:`normalize_dft`.

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

    indulge_prototypes : bool, optional
        Parameter passed to :py:func:`normalize_dft`.


    Returns
    -------
    numpy.ndarray
        an upper triangle matrix of dimension NxNx2, NxNx3 or NxNx4 holding either rgb(a)
        values corresponding to the color mapping of a single Fourier coefficient from the input, or
        the angle and magnitudes of a certain coefficient.
    """
    mag_phase_mx = normalize_dft(utm, how=magn_stra, coeff=coeff, indulge_prototypes=indulge_prototype)
    if output_raw_values:
        return mag_phase_mx
    elif magn_stra == 'raw':
        print("Raw (absolute) magnitudes cannot be converted to colors. Returning the raw "
              "magnitude-phase matrix instead.")
        return mag_phase_mx
    return circular_hue(mag_phase_mx, output_rgba, ignore_magnitude, ignore_phase, as_html=as_html)



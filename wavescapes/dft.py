import math
from functools import lru_cache
from itertools import accumulate

import numpy as np

def reset_tril(arr):
    """This is the equivalent to np.triu(arr), but only for the first two dimensions and inplace.
    This function mutates arr by overwriting the triangle beneath the diagonal (tril, k=-1)
    with zeros."""
    n = arr.shape[0]
    arr[np.tril_indices(n, k=-1)] = 0

def build_utm_from_one_row(first_row, long=False, reset_ltm=True):
    """
    given a NxM matrix representing N adjacent segments
    of a piece by M numbers each, this function creates a
    NxNxM upper triangular matrix ("triu") starting with
    the given matrix as the first row. All rows below (i >= 1)
    are created by summing for each row's element
    (i, j) the upper left neighbour (i-1, j-i) with (0, j).
    This method of summing builds an upper-triangle-matrix
    whose structure represent all hierarchical levels.

    Parameters
    ----------
    first_row: np.array
        Expects a 2D array where rows represent adjacent segments of a piece. They could be,
        for instance, beat-wise DFT coefficients or slice-wise pitch class profiles.
    long : bool, optional
        By default, the upper triangle matrix will be returned as a square matrix where the
        lower left triangle beneath the diagonal is filled with irrelevant values (see `reset_ltm`).
        Pass True to obtain the UTM in long format instead.
    reset_ltm: bool, optional
        If not `long`, the ltm (lower triangular matrix, "tril") contains irrelevant non-zero values.
        By default, these are overwritten with 0 to avoid confusion. Pass False to skip this step.

    Returns
    -------
    np.array
        (N(N+1)/2, M) if long else (N, N, M)
    """

    def shorten_previous(a, b):
        """Shift the previous row to the right by dropping the last element, and add."""
        return first_row[b:] + a[:-1]

    def pad_previous(a, _):
        """Shift the previous row to the right by prepending a zero, and add."""
        return first_row + np.pad(a, ((1, 0), (0, 0)))[:-1]

    first_row = np.atleast_2d(first_row)
    n_segments = first_row.shape[0]
    if long:
        result = np.vstack(list(accumulate(range(1, n_segments), shorten_previous, initial=first_row)))
    else:
        result = np.array(list(accumulate(range(1, n_segments), pad_previous, initial=first_row)))
        if reset_ltm:
            reset_tril(result)
    return result


def apply_dft_to_pitch_class_matrix(pc_mat, build_utm = True, long=False):
    """
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
    long : bool, optional
        By default, if `build_utm`, the upper triangle matrix will be returned as a square matrix
        where the lower left triangle beneath the diagonal is filled with zeros.
        Pass True to obtain the UTM in long format instead.
    
    Returns
    -------
    numpy matrix (numpy.ndarray of numpy.complex128)
        according to the parameters 'build_utm', either a Nx7 complex number matrix being
        the converted input matrix of pitch class distribution
        transformed into Fourier coefficient, or a NxNx7 complex number 
        upper triangle matrix being the fourier coefficient obtained from all
        possible slices of the original musical piece.
    """
    pcv_nmb, pc_nmb = np.shape(pc_mat)
    #+1 to hold room for the 0th coefficient
    coeff_nmb = int(pc_nmb/2)+1
    res = np.fft.fft(pc_mat)[:, :coeff_nmb] #coeff 7 to 11 are uninteresting (conjugates of coeff 6 to 1).
    
    if build_utm:
        res = build_utm_from_one_row(res, long=long)
        
    return res


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
        above_vals, below_vals = arr[surpassing], arr[subpassing]
        above, below = not np.allclose(above_vals, 1), not np.allclose(below_vals, 0)
        if not above and not below:
            # means the differences are due to floating point precision errors, no need to warn
            return np.clip(arr, 0, 1), ''
        fails = surpassing.sum() + subpassing.sum()
        plural = fails > 1
        msg = 'There were ' if plural else 'There was '
        if below:
            n_below = subpassing.sum()
            mean_below = np.mean(below_vals)
            msg += f"{n_below} value"
            msg += f"s < 0 (meanΔ={mean_below}) " if n_below > 1 else f" < 0 ({mean_below}) "
        if above:
            n_above = surpassing.sum()
            mean_above = np.mean(above_vals - 1.)
            if below:
                msg += 'and '
            msg += f"{n_above} value"
            msg += f"s > 1 (meanΔ={mean_above}) " if n_above > 1 else f" > 1 ({mean_above}) "
        msg += "which had to be clipped."
        return np.clip(arr, 0, 1), msg
    return arr, ''


COEFF2MAX_ANGLE = {
    1: math.pi / 6,  # 30°
    2: math.pi / 3,  # 60°
    3: math.pi / 2,  # 90°
    4: math.tau / 3,  # 120°
    5: math.pi / 6,  # 30°
}
COEFF2BETA = {
    1: 5 / 12 * math.pi,
    2: math.pi / 3,
    3: math.pi / 4,
    4: math.pi / 6,
    5: 5 / 12 * math.pi
}


@lru_cache
def phase2max_magnitude(alpha, coeff):
    """ Compute maximally possible magnitude for a given phase (angle alpha) and coefficient."""

    if coeff in (0,6) or alpha == 0:
        return 1.0
    alpha = alpha % COEFF2MAX_ANGLE[coeff]
    beta = COEFF2BETA[coeff]
    gamma = math.pi - beta - alpha
    return np.sin(beta) / np.sin(gamma)


def normalize_mag_by_phase(c, coeff):
    """Takes a complex number and normalizes the magnitude by the magnitude that
    is maximally possible for the given coefficient at that phase."""
    mag, phase = abs(c), np.angle(c)
    if coeff in (0, 6) or phase == 0:
        return np.array([mag, phase])
    max_possible_mag = phase2max_magnitude(phase, coeff)
    return np.array([mag / max_possible_mag, phase])


def max_possible_mags(complex_coeffs, coeff=None):
    """Apply this along the last axis of a matrix of (...,7) DFT coefficients."""
    cc = complex_coeffs.flatten()
    assert cc.shape[0] == 7, f"This function operates on vectors of 7 complex numbers but got shape {complex_coeffs.shape}"
    if coeff is None:
        return np.array([normalize_mag_by_phase(c, i) for i, c in enumerate(cc[1:], 1)])
    return np.array(normalize_mag_by_phase(cc[coeff], coeff))


def normalize_dft(dft=None, how='0c', coeff=None, indulge_prototypes=False):
    """ Converts complex numbers into magnitude and phase and normalizes the magnitude by one of
    several possible approaches.

    Parameters
    ----------
    dft : np.array
        An (NxNx7) upper triangular matrix of complex numbers (although the last dimension could
        be larger in principle).
    how : {'0c', 'post_norm', 'max', 'max_weighted', 'raw'}
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
    coeff : int, optional
        By default, the normalization is performed on all coefficients and an (NxNx7x2) matrix is
        returned. Pass an integer to select only one of them.
    indulge_prototypes : bool, optional
        This is an additional normalization that can be combined with any other. Since magnitudes
        of 1 are possible only for prototype phases sitting on the unit circle, you can set this
        parameter to True to normalize the magnitudes by the maximally achievable magnitude given
        the phase which is bounded by straight lines between adjacent prototypes. The pitch class
        vectors that benefit most from this normalization in terms of magnitude gain are those
        whose phase is exactly between two prototypes, such as the "octatonic" combination O₀,₁.
        The maximal "boosting" factors for the first 5 coefficients are
        {1: 1.035276, 2: 1.15470, 3: 1.30656, 4: 2.0, 5: 1.035276}. The sixth coefficient's phase
        can only be 0 or pi so it remains unchanged. Use this option if you want to compensate
        for the smaller magnitude space of the middle coefficients.

    Returns
    -------
    np.array
        (NxNx7x2) if coeff is None, else (NxNx2). In both cases the last dimension contains
        (normalized magnitude, phase) pairs.
    """
    if indulge_prototypes:
        normalized_by_max_possible_magn_given_phase = np.apply_along_axis(max_possible_mags, -1, dft, coeff)
        axes = normalized_by_max_possible_magn_given_phase.ndim - 1
        mags, phases = normalized_by_max_possible_magn_given_phase.transpose(axes, *range(axes))
    elif coeff is None:
        mags, phases = np.abs(dft[..., 1:]), np.angle(dft[..., 1:])
    else:
        mags, phases = np.abs(dft[..., coeff]), np.angle(dft[..., coeff])

    def concat_mags_phases():
        """Produce the function result by concatenating magnitudes and phases."""
        nonlocal mags, phases
        if not how == 'raw':
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
            norm_by = mags.max(axis=(1, 0), keepdims=True)
        else:
            norm_by = mags.max()
    elif how == 'max_weighted':
        if coeff is None:
            norm_by = mags.max(axis=-2, keepdims=True)
        else:
            norm_by = mags.max(axis=-1, keepdims=True)
    mags = np.divide(mags, norm_by, out=np.zeros_like(mags), where=norm_by > 0)
    if mags.ndim > 2:
        reset_tril(mags)
    if how == 'post_norm':
        if coeff is None:
            mags = mags / mags.max(axis=(1, 0), keepdims=True)
        else:
            mags = mags / mags.max()
    return concat_mags_phases()
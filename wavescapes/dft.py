import numpy as np
from itertools import accumulate

def reset_tril(arr):
    """This is the equivalent to np.triu(arr), but only for the first two dimensions and inplace.
    This function mutates arr by overwriting the triangle beneath the diagonal (tril, k=-1)
    with zeros."""
    n = arr.shape[0]
    arr[np.tril_indices(n, k=-1)] = 0

def build_utm_from_one_row(first_row, reset_ltm=True):
    """
    given a NxM matrix representing N adjacent segments
    of a piece by M numbers each, this function creates a
    NxNxM upper triangular matrix ("triu") starting with
    the given matrix as the first. All rows below (i >= 1)
    are created by summing for each row's element
    (i, j) the upper left neighbour (i-1, j-i) with (0, j).
    This method of summing builds an upper-triangle-matrix
    whose structure represent all hierarchical levels.

    Parameters
    ----------
    first_row: np.array
        Expects a 2D array where rows represent adjacent segments of a piece. They could be,
        for instance, beat-wise DFT coefficients or slice-wise pitch class profiles.
    reset_ltm: bool, optional
        The ltm (lower triangular matrix, "tril") contains irrelevant non-zero values. By default,
        these are overwritten with 0 to avoid confusion. Pass False skip this step.

    Returns
    -------
    np.array

    """
    def pad_previous(a, b):
        """Shift the previous row to the right by prepending a zero and add. ``b`` is ignored."""
        return first_row + np.pad(a, ((1, 0), (0, 0)))[:-1]
    pcv_nmb = np.shape(first_row)[0]
    result = np.array(list(accumulate(range(pcv_nmb - 1), pad_previous, initial=first_row)))
    if reset_ltm:
        reset_tril(result)
    return result


def apply_dft_to_pitch_class_matrix(pc_mat, build_utm = True):
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
        res = build_utm_from_one_row(res)
        
    return res

import numpy as np

def k_factor(s_params):
    """
    Compute Rollet's stability factor (K) for an array of 2-port S-parameters.

    Parameters
    ----------
    s_params : ndarray of shape (n_freqs, 2, 2)
        Complex S-parameter matrices at each frequency.

    Returns
    -------
    k : ndarray of shape (n_freqs,)
        Rollet's stability factor at each frequency.
    delta : ndarray of shape (n_freqs,)
        Determinant of the S-parameter matrix at each frequency.
    """
    s11 = s_params[:, 0, 0]
    s12 = s_params[:, 0, 1]
    s21 = s_params[:, 1, 0]
    s22 = s_params[:, 1, 1]

    delta = s11 * s22 - s12 * s21
    numerator = 1 - np.abs(s11)**2 - np.abs(s22)**2 + np.abs(delta)**2
    denominator = 2 * np.abs(s12 * s21)

    k = numerator / denominator
    return k, delta
import numpy as np
from scipy.signal import wavelets, convolve


def cwtmorlet(points, width):
    """complex morlet wavelet function compatible with scipy.signal.cwt
    Parameters: points: int
                    Number of points in `vector`.
                width: scalar
                    Width parameter of wavelet.
                    Equals (sample rate / fundamental frequency of wavelet)
    Returns: `vector`: complex-valued ndarray of shape (points,)
    """
    omega = 5.0
    s = points / (2.0 * omega * width)
    return wavelets.morlet(points, omega, s, complete=True)


# modified cwt for less accuracy and more speed (use 3*width, not 10*width)
def roughcwt(data, wavelet, widths):
    """
    Continuous wavelet transform.

    Performs a continuous wavelet transform on `data`,
    using the `wavelet` function. A CWT performs a convolution
    with `data` using the `wavelet` function, which is characterized
    by a width parameter and length parameter.

    Parameters
    ----------
    data : (N,) ndarray
        data on which to perform the transform.
    wavelet : function
        Wavelet function, which should take 2 arguments.
        The first argument is the number of points that the returned vector
        will have (len(wavelet(width,length)) == length).
        The second is a width parameter, defining the size of the wavelet
        (e.g. standard deviation of a gaussian). See `ricker`, which
        satisfies these requirements.
    widths : (M,) sequence
        Widths to use for transform.

    Returns
    -------
    cwt: (M, N) ndarray
        Will have shape of (len(data), len(widths)).

    Notes
    -----
    >>> length = min(10 * width[ii], len(data))
    >>> cwt[ii,:] = scipy.signal.convolve(data, wavelet(length,
    ...                                       width[ii]), mode='same')

    Examples
    --------
    >>> from scipy import signal
    >>> sig = np.random.rand(20) - 0.5
    >>> wavelet = signal.ricker
    >>> widths = np.arange(1, 11)
    >>> cwtmatr = signal.cwt(sig, wavelet, widths)

    """
    out_dtype = wavelet(widths[0], widths[0]).dtype
    output = np.zeros([len(widths), len(data)], dtype=out_dtype)
    for ind, width in enumerate(widths):
        wavelet_data = wavelet(min(3 * width, len(data)), width)
        output[ind, :] = convolve(data, wavelet_data,
                                              mode='same')
    return output

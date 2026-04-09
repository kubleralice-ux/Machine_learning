import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal.windows import gaussian
from matplotlib import colormaps
from matplotlib.colors import LogNorm
from scipy.signal import spectrogram
from scipy.signal.windows import gaussian
import soundfile as sf



def gaussian_spectrogram(x, fs, window_dur=0.02, step_dur=0.01, dyn_range=60,
                         cmap=None, ax=None):

    # set default for step_dur, if unspecified. This value is optimal for Gaussian windows.
    if step_dur is None:
        step_dur = window_dur / 2

    # convert window & step durations from seconds to numbers of samples (which is what
    # scipy.signal.spectrogram takes as input).
    window_nsamp = int(round(window_dur * fs))
    step_nsamp = int(round(step_dur * fs))

    # make the window. A Gaussian filter needs a minimum of 6σ - 1 samples
    # (https://en.wikipedia.org/wiki/Gaussian_filter#Digital_implementation),
    # so working backward from window_nsamp we can calculate σ.
    window_sigma = (window_nsamp + 1) / 6
    window = gaussian(window_nsamp, window_sigma, sym=False)

    # convert step size into number of overlapping samples in adjacent analysis frames
    noverlap = window_nsamp - step_nsamp

    # compute the power spectral density. units are V²/Hz (assuming input signal in V)
    freqs, times, power = spectrogram(
        x,
        fs=fs,
        window=window,
        nperseg=window_nsamp,
        noverlap=noverlap,
        detrend=False
    )


    p_ref = 2e-5  # 20 μPa, the standard reference pressure for sound in air

    # set lower bound of colormap (vmin) from dynamic range. The upper bound we'll set
    # as `None`, which defaults to the largest value in the spectrogram.
    dB_max = 10 * np.log10(power.max() / (p_ref ** 2))
    vmin = 10 ** ((dB_max - dyn_range) / 10) * (p_ref ** 2)
    vmax = None

    # set default colormap if none specified, then convert name to actual cmap object
    if cmap is None:
        cmap = "Greys"
    cmap = colormaps[cmap]

    # create the figure if needed
    if ax is None:
        fig, ax = plt.subplots()

    # other arguments to the figure
    extent = (times.min(), times.max(), freqs.min(), freqs.max())

    # plot
    ax.imshow(power, origin="lower", aspect="auto", cmap=cmap,
              norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent)
    return ax

if __name__ == "__main__":
    path ="biodcase_development_set/train/audio/ballenyislands2015/2015-01-15T17-00-00_000.wav"

    x, fs = sf.read(path)

    if x.ndim > 1:
        x = x[:, 0]

    print("Shape:", x.shape)
    print("Min:", x.min(), "Max:", x.max())

    ax = gaussian_spectrogram(x, fs, cmap="viridis")
    plt.show()


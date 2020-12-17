import numpy
from scipy.signal import butter, lfilter
import argparse
import multiprocessing
import itertools
from librosa.decompose import hpss
from librosa.core import stft, istft
from librosa.util import fix_length

# bark frequency bands
FREQ_BANDS = [
    20,
    119,
    224,
    326,
    438,
    561,
    698,
    850,
    1021,
    1213,
    1433,
    1685,
    1978,
    2322,
    2731,
    3227,
    3841,
    4619,
    5638,
    6938,
    8492,
    10705,
    14105,
    20000,
]


def bandpass(lo, hi, x, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return lfilter(b, a, x)


def lowpass(hi, x, fs, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, hi / nyq, btype="low")
    return lfilter(b, a, x)


def envelope(x, fs, params):
    fast_attack = params[0]
    slow_attack = params[1]
    release = params[2]
    power_mem = params[3]

    g_fast = numpy.exp(-1.0 / (fs * fast_attack / 1000.0))
    g_slow = numpy.exp(-1.0 / (fs * slow_attack / 1000.0))
    g_release = numpy.exp(-1.0 / (fs * release / 1000.0))
    g_power = numpy.exp(-1.0 / (fs * power_mem / 1000.0))

    fb_fast = 0
    fb_slow = 0
    fb_pow = 0

    N = len(x)

    fast_envelope = numpy.zeros(N)
    slow_envelope = numpy.zeros(N)
    attack_gain_curve = numpy.zeros(N)

    x_power = numpy.zeros(N)
    x_deriv_power = numpy.zeros(N)

    for n in range(N):
        x_power[n] = (1 - g_power) * x[n] * x[n] + g_power * fb_pow
        fb_pow = x_power[n]

    x_deriv_power[0] = x_power[0]

    # simple differentiator filter
    for n in range(1, N):
        x_deriv_power[n] = x_power[n] - x_power[n - 1]

    for n in range(N):
        if fb_fast > x_deriv_power[n]:
            fast_envelope[n] = (1 - g_release) * x_deriv_power[n] + g_release * fb_fast
        else:
            fast_envelope[n] = (1 - g_fast) * x_deriv_power[n] + g_fast * fb_fast
        fb_fast = fast_envelope[n]

        if fb_slow > x_deriv_power[n]:
            slow_envelope[n] = (1 - g_release) * x_deriv_power[n] + g_release * fb_slow
        else:
            slow_envelope[n] = (1 - g_slow) * x_deriv_power[n] + g_slow * fb_slow
        fb_slow = slow_envelope[n]

        attack_gain_curve[n] = fast_envelope[n] - slow_envelope[n]

    attack_gain_curve /= numpy.max(attack_gain_curve)

    # normalize to [0, 1.0]
    return x * attack_gain_curve


def single_band_transient_shaper(band, x, fs, shaper_params, order=2):
    lo = FREQ_BANDS[band]
    hi = FREQ_BANDS[band + 1]

    y = bandpass(lo, hi, x, fs)

    # per bark band, apply a differential envelope attack/transient enhancer
    y_shaped = envelope(y, fs, shaper_params)

    return y_shaped


def multiband_transient_shaper(x, fs, shaper_params, pool):
    if shaper_params[0] not in [0, 1]:
        raise ValueError("attack should be 0 (boost sustain) or 1 (boost attacks)")

    # bark band decomposition
    band_results = list(
        pool.starmap(
            single_band_transient_shaper,
            zip(
                range(0, len(FREQ_BANDS) - 1, 1),
                itertools.repeat(x),
                itertools.repeat(fs),
                itertools.repeat(shaper_params),
            ),
        )
    )

    y_t = numpy.zeros(len(x))
    for banded_attacks in band_results:
        y_t += banded_attacks

    return y_t


# iterative hpss
def ihpss(x, prog):
    # big t-f resolution for harmonic
    S1 = stft(
        x,
        n_fft=2 * prog.harmonic_frame,
        win_length=prog.harmonic_frame,
        hop_length=int(prog.harmonic_frame // 2),
    )
    S_h1, S_p1 = hpss(S1, margin=prog.harmonic_beta, power=numpy.inf)  # hard mask
    S_r1 = S1 - (S_h1 + S_p1)

    yh = fix_length(istft(S_h1, dtype=x.dtype), len(x))
    yp1 = fix_length(istft(S_p1, dtype=x.dtype), len(x))
    yr1 = fix_length(istft(S_r1, dtype=x.dtype), len(x))

    # small t-f resolution for percussive
    S2 = stft(
        yp1 + yr1,
        n_fft=2 * prog.percussive_frame,
        win_length=prog.percussive_frame,
        hop_length=int(prog.percussive_frame // 2),
    )
    _, S_p2 = hpss(S2, margin=prog.percussive_beta, power=numpy.inf)  # hard mask

    yp = fix_length(istft(S_p2, dtype=x.dtype), len(x))

    if prog.shape_transients:
        yp = multiband_transient_shaper(
            yp,
            44100,
            (
                prog.fast_attack_ms,
                prog.slow_attack_ms,
                prog.release_ms,
                prog.power_memory_ms,
            ),
            prog.pool,
        )

    return yh, yp

import numpy
from scipy.signal import butter, lfilter
import argparse
import multiprocessing
import itertools

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


def _envelope(x, fs, params):
    attack = params[0]
    fast_attack = params[1]
    slow_attack = params[2]
    release = params[3]
    power_mem = params[4]

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

    if attack == 1:
        # normalize to [0, 1.0]
        return x * attack_gain_curve

    # sustain curve is the inverse
    return x * (1 - attack_gain_curve)


def _single_band_transient_shaper(band, x, fs, shaper_params, order=2):
    nyq = 0.5 * fs

    lo = FREQ_BANDS[band]
    hi = FREQ_BANDS[band + 1]

    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    y = lfilter(b, a, x)

    # per bark band, apply a differential envelope attack/transient enhancer
    y_shaped = envelope(y, fs, shaper_params)

    return y_shaped


def _multiband_transient_shaper(x, fs, shaper_params, npool=16):
    if shaper_params[0] not in [0, 1]:
        raise ValueError("attack should be 0 (boost sustain) or 1 (boost attacks)")

    pool = multiprocessing.Pool(npool)

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

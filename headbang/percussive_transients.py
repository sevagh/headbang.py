import numpy
import itertools
from scipy.signal import butter, lfilter
from librosa.decompose import hpss
from librosa.core import stft, istft
from librosa.util import fix_length

# bark frequency bands between 20 and 20khz, human hearing stuff
_FREQ_BANDS = [
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


def _bandpass(lo, hi, x, fs, order):
    nyq = 0.5 * fs
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return lfilter(b, a, x)


def _attack_envelope(x, fs, fast_attack_ms, slow_attack_ms, release_ms, power_memory_ms):
    g_fast = numpy.exp(-1.0 / (fs * fast_attack_ms / 1000.0))
    g_slow = numpy.exp(-1.0 / (fs * slow_attack_ms / 1000.0))
    g_release = numpy.exp(-1.0 / (fs * release_ms / 1000.0))
    g_power = numpy.exp(-1.0 / (fs * power_memory_ms / 1000.0))

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


def _single_band_transient_shaper(band, x, fs, fast_attack_ms, slow_attack_ms, release_ms, power_memory_ms, filter_order):
    lo = _FREQ_BANDS[band]
    hi = _FREQ_BANDS[band + 1]

    y = _bandpass(lo, hi, x, fs, filter_order)

    # per bark band, apply a differential envelope attack/transient enhancer
    y_shaped = _attack_envelope(y, fs, fast_attack_ms, slow_attack_ms, release_ms, power_memory_ms)

    return y_shaped


def _multiband_transient_shaper(x, fs, pool, fast_attack_ms, slow_attack_ms, release_ms, power_memory_ms, filter_order):
    # bark band decomposition
    band_results = list(
        pool.starmap(
            _single_band_transient_shaper,
            zip(
                range(0, len(_FREQ_BANDS) - 1, 1),
                itertools.repeat(x),
                itertools.repeat(fs),
                itertools.repeat(fast_attack_ms),
                itertools.repeat(slow_attack_ms),
                itertools.repeat(release_ms),
                itertools.repeat(power_memory_ms),
                itertools.repeat(filter_order),
            ),
        )
    )

    y_t = numpy.zeros(len(x))
    for banded_attacks in band_results:
        y_t += banded_attacks

    return y_t


# iterative hpss
def ihpss(x, pool, harmonic_margin=2.0, harmonic_frame=4096, percussive_margin=2.0, percussive_frame=256,
        fast_attack_ms=1, slow_attack_ms=15, release_ms=20, power_memory_ms=1, filter_order=2):
    print(
        "Iteration 1 of hpss: frame = {0}, margin = {1}".format(
            harmonic_frame, harmonic_margin
        )
    )
    # big t-f resolution for harmonic
    S1 = stft(
        x,
        n_fft=2 * harmonic_frame,
        win_length=harmonic_frame,
        hop_length=int(harmonic_frame // 2),
    )
    S_h1, S_p1 = hpss(S1, margin=harmonic_margin, power=numpy.inf)  # hard mask
    S_r1 = S1 - (S_h1 + S_p1)

    yh = fix_length(istft(S_h1, dtype=x.dtype), len(x))
    yp1 = fix_length(istft(S_p1, dtype=x.dtype), len(x))
    yr1 = fix_length(istft(S_r1, dtype=x.dtype), len(x))

    print(
        "Iteration 2 of hpss: frame = {0}, margin = {1}".format(
            percussive_frame, percussive_margin
        )
    )
    # small t-f resolution for percussive
    S2 = stft(
        yp1 + yr1,
        n_fft=2 * percussive_frame,
        win_length=percussive_frame,
        hop_length=int(percussive_frame // 2),
    )
    _, S_p2 = hpss(S2, margin=percussive_margin, power=numpy.inf)  # hard mask

    yp = fix_length(istft(S_p2, dtype=x.dtype), len(x))

    print("Applying multiband transient shaper:\n\tfast attack (ms) = {0},\n\tslow attack (ms) = {1},\n\trelease (ms) = {2},\n\tpower memory (ms) = {3},\n\tfilter order = {4}".format(
        fast_attack_ms,
        slow_attack_ms,
        release_ms,
        power_memory_ms,
        filter_order,
    ))
    yp_tshaped = _multiband_transient_shaper(
        yp,
        44100,
        pool,
        fast_attack_ms,
        slow_attack_ms,
        release_ms,
        power_memory_ms,
        filter_order,
    )

    return yp_tshaped, yp

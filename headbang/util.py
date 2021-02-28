from madmom.io.audio import load_audio_file
import numpy
import librosa


def load_wav(wav_in, stereo=False):
    x, fs = load_audio_file(wav_in, sample_rate=44100)

    if not stereo:
        # stereo to mono if necessary
        if len(x.shape) > 1 and x.shape[1] == 2:
            x = x.sum(axis=1) / 2

    # cast to float
    x = x.astype(numpy.single)

    # normalize between -1.0 and 1.0
    x /= numpy.max(numpy.abs(x))

    return x


def overlay_clicks(x, beats):
    clicks = librosa.clicks(beats, sr=44100, length=len(x))

    if len(x.shape) > 1 and x.shape[1] == 2:
        clicks = numpy.column_stack((clicks, clicks))  # convert to stereo

    return (x + clicks).astype(numpy.single)


def find_closest(A, target):
    idx = A.searchsorted(target)
    idx = numpy.clip(idx, 1, len(A) - 1)
    left = A[idx - 1]
    right = A[idx]
    idx -= target - left < right - target
    return idx

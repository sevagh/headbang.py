import numpy
import itertools
from essentia.standard import (
    OnsetDetection,
    Onsets,
    Windowing,
    FFT,
    CartesianToPolar,
    FrameGenerator,
    BeatTrackerMultiFeature,
    BeatTrackerDegara,
)
from essentia import Pool, array

"""
these have to be global because they're not pickleable otherwise
and can't be parallelized
"""

ODF = ["hfc", "rms"]
_ONSET_DETECTORS = [OnsetDetection(method=f) for f in ODF]

_w = Windowing(type="hann")
_fft = FFT()
_c2p = CartesianToPolar()


def _apply_single_odf(odf_idx, frame):
    (
        mag,
        phase,
    ) = _c2p(_fft(_w(frame.astype(numpy.single))))
    return _ONSET_DETECTORS[odf_idx](mag, phase)


class OnsetDetector:
    def __init__(self, silence_threshold):
        self.pool = Pool()
        self.onsets = Onsets(silenceThreshold=silence_threshold)

        weights = numpy.ones(len(ODF))
        weights[0] = 4.0  # weight hfc stronger

        self.weights = weights.astype(numpy.single)

        self.frame_size = 1024
        self.hop_size = 512

    def detect_onsets(self, x, pool):
        # Computing onset detection functions.
        for frame in FrameGenerator(
            x.astype(numpy.single), frameSize=self.frame_size, hopSize=self.hop_size
        ):
            onset_features = pool.starmap(
                _apply_single_odf,
                zip(
                    range(len(ODF)),
                    itertools.repeat(frame),
                ),
            )
            for i, of in enumerate(onset_features):
                self.pool.add("features.{0}".format(ODF[i]), of)

        # convert pool into matrix
        matrix = [
            array(self.pool["features.{0}".format(ODF[i])]) for i in range(len(ODF))
        ]

        return self.onsets(matrix, self.weights)

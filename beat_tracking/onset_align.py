from essentia.standard import (
    OnsetDetection,
    Onsets,
    Windowing,
    FFT,
    CartesianToPolar,
    FrameGenerator,
)
import essentia
import itertools
import numpy

ODF = ["hfc", "complex", "flux", "rms"]
ONSET_DETECTORS = [OnsetDetection(method=f) for f in ODF]

w = Windowing(type="hann")
fft = FFT()
c2p = CartesianToPolar()


def apply_single_odf(odf_idx, frame):
    (
        mag,
        phase,
    ) = c2p(fft(w(frame)))
    return ONSET_DETECTORS[odf_idx](mag, phase)


class OnsetAligner:
    def __init__(self, silence_threshold):
        # cribbed straight from the essentia examples
        # https://essentia.upf.edu/essentia_python_examples.html
        self.pool = essentia.Pool()

        self.onsets = Onsets(silenceThreshold=silence_threshold)

        # evenly weight all ODF functions for now
        self.weights = numpy.ones(len(ODF))

    def align_beats(self, beats, prog):
        # Computing onset detection functions.
        for frame in FrameGenerator(prog.xp, frameSize=1024, hopSize=512):
            onset_features = prog.pool.starmap(
                apply_single_odf,
                zip(
                    range(len(ODF)),
                    itertools.repeat(frame),
                ),
            )
            for i, of in enumerate(onset_features):
                self.pool.add("features.{0}".format(ODF[i]), of)

        # convert pool into matrix
        matrix = [
            essentia.array(self.pool["features.{0}".format(ODF[i])])
            for i in range(len(ODF))
        ]

        total_onsets = self.onsets(matrix, self.weights)

        i = 0
        j = 0

        aligned_beats = []
        while i < len(total_onsets) and j < len(beats):
            curr_onset = total_onsets[i]
            curr_beat = beats[j]

            if numpy.abs(curr_onset - curr_beat) <= prog.beat_near_threshold:
                # always use the onset, not the beat
                aligned_beats.append(curr_onset)
                i += 1
                j += 1
                continue

            if curr_beat < curr_onset:
                # increment beats
                j += 1
            elif curr_beat > curr_onset:
                i += 1

        return aligned_beats

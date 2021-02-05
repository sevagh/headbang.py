from .percussive_transients import ihpss
from .onset import OnsetDetector, ODF
from .beattrack import ConsensusBeatTracker
import numpy


def align_beats_onsets(beats, onsets, thresh):
    i = 0
    j = 0

    aligned_beats = []
    time_since_last_beat = 0.0

    while i < len(onsets) and j < len(beats):
        curr_onset = onsets[i]
        curr_beat = beats[j]

        if numpy.abs(curr_onset - curr_beat) <= thresh:
            aligned_beats.append((curr_onset + curr_beat) / 2)
            i += 1
            j += 1
            continue

        if curr_beat < curr_onset:
            # increment beats
            j += 1
        elif curr_beat > curr_onset:
            i += 1

    return aligned_beats


class HeadbangBeatTracker:
    def __init__(
        self,
        pool,
        # consensus beat tracking params
        algorithms="1,2,3,4,5,6,7,8",
        beat_near_threshold_s=0.1,
        consensus_ratio=0.5,
        # onset alignment params
        disable_onsets=False,
        max_no_beats=2.0,
        onset_near_threshold_s=0.1,
        onset_silence_threshold=0.035,
        # hpss params
        harmonic_margin=2.0,
        harmonic_frame=4096,
        percussive_margin=2.0,
        percussive_frame=256,
        # transient shaper params
        fast_attack_ms=1,
        slow_attack_ms=15,
        release_ms=20,
        power_memory_ms=1,
        filter_order=2,
    ):
        self.beat_near_threshold_s = beat_near_threshold_s
        self.max_no_beats = max_no_beats
        self.onset_near_threshold_s = onset_near_threshold_s

        self.pool = pool
        self.cbt = ConsensusBeatTracker(
            self.pool,
            algorithms=algorithms,
            beat_near_threshold_s=beat_near_threshold_s,
            consensus_ratio=consensus_ratio,
        )

        self.cbt.print_params()
        self.onset_detector = OnsetDetector(onset_silence_threshold)
        self.disable_onsets = disable_onsets

        self.harmonic_margin = harmonic_margin
        self.harmonic_frame = harmonic_frame
        self.percussive_margin = percussive_margin
        self.percussive_frame = percussive_frame

        self.fast_attack_ms = fast_attack_ms
        self.slow_attack_ms = slow_attack_ms
        self.release_ms = release_ms
        self.power_memory_ms = power_memory_ms
        self.filter_order = filter_order

    def beats(self, x):
        self.beat_consensus = self.cbt.beats(x)
        if self.disable_onsets:
            print("Onset alignment disabled, returning consensus beats")
            return self.beat_consensus

        print(
            "Creating percussive separation with enhanced transients for percussive onset detection"
        )

        # get a percussive separation for onset alignment, and the percussive spectrum
        self.xp, self.xp_hpss = ihpss(
            x,
            self.pool,
            self.harmonic_margin,
            self.harmonic_frame,
            self.percussive_margin,
            self.percussive_frame,
            self.fast_attack_ms,
            self.slow_attack_ms,
            self.release_ms,
            self.power_memory_ms,
            self.filter_order,
        )

        print("Detecting percussive onsets with methods {0}".format(ODF))
        self.onsets = self.onset_detector.detect_onsets(self.xp, self.pool)

        print("Aligning agreed beats with percussive onsets")
        self.aligned = align_beats_onsets(
            self.beat_consensus, self.onsets, self.beat_near_threshold_s
        )

        print("Trying to substitute percussive onsets in place of absent beats")
        # add a 0 in there in case no beats have been found until the first, very deep into the song
        # also concatenate the max length for that case too
        endofsong = (len(x) - 1) / 44100.0

        aligned_prime = numpy.concatenate(([0.0], self.aligned, [endofsong]))

        beat_jumps = numpy.where(numpy.diff(aligned_prime) > self.max_no_beats)[0]

        self.to_concat = numpy.array([])

        # collect extra beats by applying consensus beat tracking specifically to low-information segments
        for j in beat_jumps:
            try:
                print(
                    "segment with no beats: {0}-{1}".format(
                        aligned_prime[j], aligned_prime[j + 1]
                    )
                )

                segment_onsets = self.onsets[
                    numpy.where(
                        numpy.logical_and(
                            self.onsets > aligned_prime[j] + 1.0,
                            self.onsets < aligned_prime[j + 1] - 1.0,
                        )
                    )[0]
                ]

                sparse_onsets = numpy.split(
                    segment_onsets,
                    numpy.where(
                        numpy.diff(segment_onsets) > self.onset_near_threshold_s
                    )[0]
                    + 1,
                )

                so = [s[0] for s in sparse_onsets if s.size > 0]

                if so:
                    print(
                        "supplementing with percussive onsets from this region: {0}".format(
                            so
                        )
                    )
                    self.to_concat = numpy.concatenate((self.to_concat, so))
            except IndexError:
                break

        self.aligned = numpy.sort(numpy.concatenate((self.aligned, self.to_concat)))

        return self.aligned

from .beat_tracking import apply_single_beat_tracker
from .effects import ihpss, bandpass, lowpass
import numpy
import itertools
from essentia.standard import FrameGenerator


def get_consensus_beats(all_beats, max_consensus, prog):
    # no point getting a consensus of a single algorithm
    if max_consensus == 1:
        return all_beats

    good_beats = numpy.sort(numpy.unique(all_beats))
    grouped_beats = numpy.split(
        good_beats,
        numpy.where(numpy.diff(good_beats) ** 2 > prog.beat_near_threshold)[0] + 1,
    )

    beats = [x[0] for x in grouped_beats]
    beats = [numpy.mean(x) for x in grouped_beats]

    tick_agreements = [len(x) for x in grouped_beats]

    final_beats = []
    for i, tick in enumerate(beats):
        # at least CONSENSUS beat trackers agree
        if tick_agreements[i] > max_consensus * prog.consensus_ratio:
            final_beats.append(tick)

    return final_beats


# apply beat tracking to the song segmented into chunks
def chunked_algorithm(prog, chunk_seconds):
    chunk_samples = int(chunk_seconds * 44100.0)

    # need a consensus across all algorithms
    all_beats = numpy.array([])
    nframes = 0

    for frame in FrameGenerator(prog.x, frameSize=chunk_samples, hopSize=chunk_samples):
        beat_results = prog.pool.starmap(
            apply_single_beat_tracker,
            zip(
                itertools.repeat(prog.x),
                prog.beat_tracking_algorithms,
                itertools.repeat(nframes * chunk_seconds),
            ),
        )

        for beats in beat_results:
            all_beats = numpy.concatenate((all_beats, beats))

        nframes += 1

    all_beats = numpy.sort(all_beats)

    return get_consensus_beats(all_beats, len(prog.beat_tracking_algorithms), prog)


def apply_meta_algorithm(prog):
    #######################
    # pass 1 - whole song #
    #######################

    # gather all the beats from all beat tracking algorithms
    beat_results = prog.pool.starmap(
        apply_single_beat_tracker,
        zip(itertools.repeat(prog.x), prog.beat_tracking_algorithms),
    )

    # need a consensus across all algorithms
    all_beats = numpy.array([])

    for beats in beat_results:
        all_beats = numpy.concatenate((all_beats, beats))

    #####################
    # pass 2 - segments #
    #####################

    #beats_segment_10s = chunked_algorithm(prog, 10.0)
    #beats_segment_5s = chunked_algorithm(prog, 5.0)
    #beats_segment_2s = chunked_algorithm(prog, 2.0)

    ## add the segmented beats
    #all_beats = numpy.concatenate((all_beats, beats_segment_10s))
    #all_beats = numpy.concatenate((all_beats, beats_segment_5s))
    #all_beats = numpy.concatenate((all_beats, beats_segment_2s))

    all_beats = numpy.sort(all_beats)

    # 2 sets of beats, maximum of 4xalgo consensus (with some extra from the chunks)
    beat_consensus = get_consensus_beats(
        all_beats, len(prog.beat_tracking_algorithms), prog
    )

    return beat_consensus

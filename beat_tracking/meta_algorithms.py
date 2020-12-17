from .beat_tracking import apply_single_beat_tracker
from .effects import ihpss
import numpy
import itertools


def get_consensus_beats(all_beats, max_consensus, prog):
    good_beats = numpy.sort(numpy.unique(all_beats))
    grouped_beats = numpy.split(good_beats, numpy.where(numpy.diff(good_beats)**2 > prog.beat_near_threshold)[0]+1)

    beats = [x[0] for x in grouped_beats]
    beats = [numpy.mean(x) for x in grouped_beats]

    tick_agreements = [len(x) for x in grouped_beats]

    final_beats = []
    for i, tick in enumerate(beats):
        # at least CONSENSUS beat trackers agree
        if tick_agreements[i] > max_consensus*prog.consensus_ratio:
            final_beats.append(tick)

    return final_beats


def percussive(prog):
    _, x = ihpss(prog.x, prog)

    if len(prog.beat_tracking_algorithms) == 1:
        return _ALGOS[prog.beat_tracking_algorithms[0]](x)
    else:
        # need a consensus across many
        all_beats = numpy.array([])

        # gather all the beats from all beat tracking algorithms
        beat_results = prog.pool.starmap(
            apply_single_beat_tracker,
            zip(
                itertools.repeat(x),
                prog.beat_tracking_algorithms
            ),
        )

        for beats in beat_results:
            all_beats = numpy.concatenate((all_beats, beats))

        all_beats = numpy.sort(all_beats)

        return get_consensus_beats(all_beats, len(prog.beat_tracking_algorithms), prog)


def basic(prog):
    if len(prog.beat_tracking_algorithms) == 1:
        return _ALGOS[prog.beat_tracking_algorithms[0]](x)
    else:
        # need a consensus across many
        all_beats = numpy.array([])

        # gather all the beats from all beat tracking algorithms
        beat_results = prog.pool.starmap(
            apply_single_beat_tracker,
            zip(
                itertools.repeat(prog.x),
                prog.beat_tracking_algorithms
            ),
        )

        for beats in beat_results:
            all_beats = numpy.concatenate((all_beats, beats))

        all_beats = numpy.sort(all_beats)

        return get_consensus_beats(all_beats, len(prog.beat_tracking_algorithms), prog)


_META_ALGOS = {
        1: basic,
        2: percussive,
}


def apply_meta_algorithm(prog):
    return _META_ALGOS[prog.meta_algorithm](prog)

from .beat_tracking import apply_single_beat_tracker
from .effects import ihpss, bandpass, lowpass
import numpy
import itertools


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


def kick_snare(prog):
    # 0-120hz for kick drum
    kick = lowpass(120, prog.x, 44100)

    # 200-500hz for snare
    snare = bandpass(200, 500, prog.x, 44100)

    _, xpk = ihpss(kick, prog)
    _, xps = ihpss(snare, prog)

    # need a consensus across all algorithms
    all_beats = numpy.array([])

    # gather all the beats from all beat tracking algorithms
    beat_results = prog.pool.starmap(
        apply_single_beat_tracker,
        itertools.product([xpk, xps], prog.beat_tracking_algorithms),
    )

    for beats in beat_results:
        all_beats = numpy.concatenate((all_beats, beats))

    all_beats = numpy.sort(all_beats)

    # consensus across all algos * 2 (kick + snare)
    return get_consensus_beats(all_beats, 2 * len(prog.beat_tracking_algorithms), prog)


def percussive(prog):
    _, xp = ihpss(prog.x, prog)
    prog.xp = xp

    # need a consensus across all algorithms
    all_beats = numpy.array([])

    # gather all the beats from all beat tracking algorithms
    beat_results = prog.pool.starmap(
        apply_single_beat_tracker,
        itertools.product([xp], prog.beat_tracking_algorithms),
    )

    for beats in beat_results:
        all_beats = numpy.concatenate((all_beats, beats))

    all_beats = numpy.sort(all_beats)

    return get_consensus_beats(all_beats, len(prog.beat_tracking_algorithms), prog)


def basic(prog):
    # need a consensus across all algorithms
    all_beats = numpy.array([])

    # gather all the beats from all beat tracking algorithms
    beat_results = prog.pool.starmap(
        apply_single_beat_tracker,
        zip(itertools.repeat(prog.x), prog.beat_tracking_algorithms),
    )

    for beats in beat_results:
        all_beats = numpy.concatenate((all_beats, beats))

    all_beats = numpy.sort(all_beats)

    return get_consensus_beats(all_beats, len(prog.beat_tracking_algorithms), prog)


META_ALGOS = {
    1: basic,
    2: percussive,
    3: kick_snare,
}


def apply_meta_algorithm(prog):
    return META_ALGOS[prog.meta_algorithm](prog)

import numpy


def get_consensus_beats(all_beats, max_consensus, beat_near_threshold, consensus_ratio, tick_strategy):
    good_beats = numpy.sort(numpy.unique(all_beats))
    grouped_beats = numpy.split(good_beats, numpy.where(numpy.diff(good_beats)**2 > beat_near_threshold)[0]+1)

    if tick_strategy == 'first':
        beats = [x[0] for x in grouped_beats]
    elif tick_strategy == 'mean':
        beats = [numpy.mean(x) for x in grouped_beats]
    elif tick_strategy == 'median':
        beats = [numpy.median(x) for x in grouped_beats]
    else:
        raise ValueError('invalid tick_strategy {0}'.format(tick_strategy))

    tick_agreements = [len(x) for x in grouped_beats]

    final_beats = []
    for i, tick in enumerate(beats):
        # at least CONSENSUS beat trackers agree
        if tick_agreements[i] > max_consensus*consensus_ratio:
            final_beats.append(tick)

    return final_beats

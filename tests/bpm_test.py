#!/usr/bin/env python

import numpy
import scipy
from scipy import stats

import sys


def bpm_from_beats(beats):
    if beats.size == 0:
        return 0
    m_res = scipy.stats.linregress(numpy.arange(len(beats)), beats)

    first_beat = m_res.intercept
    beat_step = m_res.slope

    return 60 / beat_step


if __name__ == "__main__":
    duration = 10
    bpms = [60, 72, 75, 83, 95, 113, 152]

    for bpm in bpms:
        step = 60 / bpm
        bop_times = numpy.arange(0, duration, step)

        print(bpm_from_beats(bop_times))

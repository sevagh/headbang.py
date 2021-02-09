#!/usr/bin/env python

import os
import numpy
import multiprocessing
import argparse
import sys
from tqdm import tqdm
import librosa
from madmom.io.audio import load_audio_file, write_wave_file
import madmom
from headbang.beattrack import ConsensusBeatTracker
from collections import defaultdict
import mir_eval.beat as mir_eval_beat
from mir_eval import util
from mir_eval.io import load_events as mir_eval_txt_beats
from tabulate import tabulate
import itertools

# audio MUST be loaded at 44100

# modification of https://github.com/craffel/mir_eval/blob/master/mir_eval/beat.py
def precision_recall(reference_beats, estimated_beats, f_measure_threshold=0.07):
    mir_eval_beat.validate(reference_beats, estimated_beats)
    # When estimated beats are empty, no beats are correct; metric is 0
    if estimated_beats.size == 0 or reference_beats.size == 0:
        return (0.0, 0.0)
    # Compute the best-case matching between reference and estimated locations
    matching = util.match_events(reference_beats, estimated_beats, f_measure_threshold)

    precision = float(len(matching)) / len(estimated_beats)
    recall = float(len(matching)) / len(reference_beats)
    return precision, recall


"""
`mir_eval.beat.f_measure`: The F-measure of the beat sequence, where an
  estimated beat is considered correct if it is sufficiently close to a
  reference beat
`mir_eval.beat.cemgil`: Cemgil's score, which computes the sum of
  Gaussian errors for each beat
`mir_eval.beat.goto`: Goto's score, a binary score which is 1 when at
  least 25\% of the estimated beat sequence closely matches the reference beat
  sequence
`mir_eval.beat.p_score`: McKinney's P-score, which computes the
  cross-correlation of the estimated and reference beat sequences represented
  as impulse trains
"""

pool = multiprocessing.Pool(16)

cbts = []

algos = [1,2,3,4,5,6]
algo_combos = []

for i in range(len(algos)):
    for algo_combo in itertools.combinations(algos, i+1):
        algo_combos.append(','.join([str(x) for x in algo_combo]))

for ag in algo_combos:
    cbts.append(ConsensusBeatTracker(pool, algorithms=ag))


def eval_beats(signal, ground_truth):
    proc_beat = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act_beat = madmom.features.beats.RNNBeatProcessor()(signal)

    beat_times_1 = proc_beat(act_beat)

    n_algos = len(cbts)+1

    ret = numpy.zeros(dtype=numpy.float32, shape=(n_algos, 6))

    gt_trimmed = mir_eval_beat.trim_beats(ground_truth)

    beat_times = [mir_eval_beat.trim_beats(beat_times_1)] + [mir_eval_beat.trim_beats(cbt.beats(signal)) for cbt in cbts]

    for i, beats in enumerate(beat_times):
        ret[i][0] = mir_eval_beat.f_measure(gt_trimmed, beats)
        ret[i][1] = mir_eval_beat.cemgil(gt_trimmed, beats)[0]
        ret[i][2] = mir_eval_beat.goto(gt_trimmed, beats)
        ret[i][3] = mir_eval_beat.p_score(gt_trimmed, beats)
        (ret[i][4], ret[i][5]) = precision_recall(gt_trimmed, beats)

    return ret


def load_wav_mono_44100(wav_in):
    x, fs = load_audio_file(wav_in, sample_rate=44100)
    # stereo to mono if necessary
    if len(x.shape) > 1 and x.shape[1] == 2:
        x = x.sum(axis=1) / 2
    # cast to float
    x = x.astype(numpy.single)
    # normalize between -1.0 and 1.0
    x /= numpy.max(numpy.abs(x))
    return x


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the consensus beat tracker against the SMC dataset"
    )
    parser.add_argument("smc_dir", type=str, help="path to SMC dataset directory")

    args = parser.parse_args()

    wav_gt = defaultdict(dict)

    for elem in os.scandir(args.smc_dir):
        if "Audio" in elem.name:
            # these are the wav files
            for dir_name, _, file_list in os.walk(elem):
                for f in file_list:
                    prefix = f.split(".")[0]
                    wav_gt[prefix]["wav"] = os.path.join(dir_name, f)
        elif "Annotations_05_08_2014" in elem.name:  # adjusted/best annotations
            # these are the ground truths
            for dir_name, _, file_list in os.walk(elem):
                for f in file_list:
                    prefix = "_".join(f.split(".")[0].split("_")[:2])
                    wav_gt[prefix]["groundtruth"] = os.path.join(dir_name, f)

    total_results = numpy.zeros(dtype=numpy.float, shape=(len(wav_gt), len(cbts)+1, 6))

    seq = 0
    for item, wav_gt_pair in tqdm(wav_gt.items()):
        # load the audio
        x = load_wav_mono_44100(wav_gt_pair["wav"])

        ground_truth = mir_eval_txt_beats(wav_gt_pair["groundtruth"])

        # append ndarray 2x6 with each of the metrics for 2 algos, ref bock dbn and consensus
        total_results[seq, :, :] = eval_beats(x, ground_truth)
        seq += 1

    headers = [
        "algorithm",
        "F-measure",
        "Cemgil",
        "Goto",
        "McKinney P-score",
        "Precision",
        "Recall",
    ]
    algos = ["SB1"]
    for i, cbt in enumerate(cbts):
        algos.append("consensus"+cbt.algorithms)

    table = []

    for i, algo in enumerate(algos):
        table.append(
            [
                algo,
                numpy.mean(total_results[:, i, 0]),
                numpy.mean(total_results[:, i, 1]),
                numpy.mean(total_results[:, i, 2]),
                numpy.mean(total_results[:, i, 3]),
                numpy.mean(total_results[:, i, 4]),
                numpy.mean(total_results[:, i, 5]),
            ]
        )

    print(tabulate(table, headers, tablefmt="github"))


if __name__ == "__main__":
    main()

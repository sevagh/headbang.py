# consensus beat tracking

The code for the consensus beat tracker is in [beattrack.py](./groove_dashboard/beattrack.py). This is a short writeup describing the implementation of the algorithm, as well as some evaluation results.

## Overview

Here's a short demonstration with plots on how the consensus beat tracker works, starting from the input waveform:

![input_waveform](./.github/input_waveform.png)

8 beat trackers are applied on the input directly (without preprocessing):

![input_waveform_all_beats](./.github/input_waveform_all_beats.png)

The list of beat trackers is:
1. [madmom](https://madmom.readthedocs.io/en/latest/modules/features/beats.html) RNNBeatProcessor -> DBNBeatTrackingProcessor
2. madmom RNNBeatProcessor -> BeatTrackingProcessor
3. madmom RNNBeatProcessor -> CRFBeatDetectionProcessor
4. madmom RNNBeatProcessor -> BeatDetectionProcessor
5. [Essentia BeatTrackerMultiFeature](https://essentia.upf.edu/reference/std_BeatTrackerMultiFeature.html)
6. [Essentia BeatTrackerDegara](https://essentia.upf.edu/reference/std_BeatTrackerDegara.html)
7. [librosa beat_track](https://librosa.org/doc/latest/generated/librosa.beat.beat_track.html)
8. [BTrack](https://github.com/adamstark/BTrack)

These are executed in parallel using Python's [multiprocessing](https://docs.python.org/3/library/multiprocessing.html) module. The code is designed to be simply executed by the pool [`starmap`](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.starmap) pool function:
```python
def apply_single_beattracker(x, algo):
    beats = None
    if algo == 1:
        beats = algo1(x, ...)
    elif algo == 2:
        beats = algo2(x, ...)
    ...

    return beats

def apply_consensus(x):
    all_algos = [1, 2, 3, 4, 5, 6, 7, 8]
    all_results = pool.starmap(
            apply_single_beattracker,
            zip(itertools.repeat(x), all_algos)
    )
```

This results in the following executions invoked in parallel on the input signal `x`:
```python
apply_single_beattracker(x, 1)
apply_single_beattracker(x, 2)
...
```

Finally, all the results are accumulated in one sorted Numpy ndarray.

A true beat is marked when 5 out of the 8 beat trackers agree (i.e. `--consensus-ratio=0.6`x`len(algorithms)` predict a beat within the same `--beat-near-threshold`, by default 100ms):

![input_waveform_beat_consensus](.github/input_waveform_beat_consensus.png)

The final consensus algorithm is achieved concisely with numpy:

```python
# all_beats contains the beat locations output by all 8 beat trackers

good_beats = numpy.sort(numpy.unique(all_beats))

# group beats into buckets that are near each other by beat_near_threshold
grouped_beats = numpy.split(
    good_beats,
    numpy.where(numpy.diff(good_beats) > beat_near_threshold)[0] + 1,
)

# use the mean of each bucket as the final beat location of the bucket
beats = [numpy.mean(x) for x in grouped_beats]

# count the count of each bucket
tick_agreements = [len(x) for x in grouped_beats]

# include beats that have at least consensus agreements
final_beats = []
for i, tick in enumerate(beats):
    if tick_agreements[i] >= consensus:
        final_beats.append(tick)

return final_beats
```

## Post-processing with percussive onset alignment

### Avoiding false positives

The consensus beat tracker is focused on avoiding false positives, at the expense of potentially missing real beats. Why? I personally prefer no output over an incorrect beat location. Nothing is more satisfying than hearing a click on a beat, and nothing is worse than hearing the next click land on something that's not a beat. Silence is better.

One way of verifying the claim that the consensus beat tracker avoids false positives is to consider the two constituents of the F-measure - precision and recall.

From the definition of mir_eval's F-measure implementation:
```python
precision = float(len(matching))/len(estimated_beats)
recall = float(len(matching))/len(reference_beats)
```

If the hypothesis is that consensus beat tracker let us avoid false positives that SB1 would suffer from, then one would expect a better precision but worse recall:
```python
precision_sb1 = float(len(matching_sb1))/len(estimated_sb1)
precision_consensus = float(len(matching_consensus))/len(estimated_consensus)

recall_sb1 = float(len(matching_sb1))/len(reference_beats)
precision_consensus = float(len(matching_consensus))/len(reference_beats)
```

We can verify this by adding precision and recall separately to the final list of evaluation metrics.

## Evaluation

There's a small testbench for evaluating the consensus beat tracker against the MIREX SMC12 dataset[[1]](#1) and the [mir_eval](https://github.com/craffel/mir_eval) library[[2]](#2).

[MIREX 2019](https://www.music-ir.org/mirex/wiki/2019:MIREX2019_Results) is the most recent year of the audio beat tracking challenge (2020 results are not ready yet).

The summary of results on the SMC dataset is:
![mirex19](./.github/mirex19.png)

To anchor my own evaluation to the above, I will include results for the consensus beat tracker alongside the madmom [DBNBeatTracker](https://github.com/CPJKU/madmom/blob/master/bin/DBNBeatTracker)[[3]](#3), or SB1 in the above table. Note that this beat tracker is among the 8 used in my consensus algorithm.

## Results

The [mir_beat_eval.py](./mir_beat_eval.py) script loads the SMC dataset (which you can [download here](http://smc.inesctec.pt/research/data-2/)), which contains the ground-truth annotations. It then evaluates the results of the consensus beat tracker and the madmom DBNBeatTracker. The median score for each measure was taken across the 218 tracks of SMC, with 4 measures from mir_eval. The 4 measures (F-measure, Cemgil, Goto, and McKinney P-Score) are the same as those used in MIREX, and are borrowed from the [Beat Evaluation Toolbox](https://code.soundsoftware.ac.uk/projects/beat-evaluation/)[[4]](#4).

Testbench invocation, showing a progress bar over the 217 iterations (for SMC's 218 tracks):
```python

(groove-dash) sevagh:groove-dashboard $ ./mir_beat_eval.py ~/TRAINING-MUSIC/beat-tracking-datasets/SMC_MIREX/
[   INFO   ] MusicExtractorSVM: no classifier models were configured by default
  2%|█▋                                                                                        | 4/217 [01:04<57:07, 16.09s/it]
```
Output:

```
| algorithm   |   F-measure |   Cemgil |      Goto |   McKinney P-score |
|-------------|-------------|----------|-----------|--------------------|
| SB1         |    0.55288  | 0.436248 | 0.225806  |           0.649583 |
| consensus   |    0.461467 | 0.36056  | 0.0368664 |           0.527321 |
```

## Discussion

 and repeating the testbench:

## References

<a id="1">[1]</a>
Holzapfel, A.; Davies, M.E.P.; Zapata, J.R.; Oliveira, J.L.; Gouyon, F.; , "Selective Sampling for Beat Tracking Evaluation," Audio, Speech, and Language Processing, IEEE Transactions on , vol.20, no.9, pp.2539-2548, Nov. 2012
doi: 10.1109/TASL.2012.2205244
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6220849&isnumber=6268383

<a id="2">[2]</a>
Böck, Sebastian & Krebs, Florian & Widmer, Gerhard. (2014). A MULTI-MODEL APPROACH TO BEAT TRACKING CONSIDERING HETEROGENEOUS MUSIC STYLES.

<a id="3">[3]</a>
Colin Raffel, Brian McFee, Eric J. Humphrey, Justin Salamon, Oriol Nieto, Dawen Liang, and Daniel P. W. Ellis, "mir_eval: A Transparent Implementation of Common MIR Metrics", Proceedings of the 15th International Conference on Music Information Retrieval, 2014.

<a id="4">[4]</a>
Matthew E. P. Davies,  Norberto Degara, and Mark D. Plumbley. "Evaluation Methods for Musical Audio Beat Tracking Algorithms", Queen Mary University of London Technical Report C4DM-TR-09-06, London, United Kingdom, 8 October 2009.

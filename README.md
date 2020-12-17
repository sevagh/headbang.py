# beat-tracking

A wacky beat-tracking toolbox.

The goal of the beat-tracking project is to house some useful or experimental variants of accurate beat tracking for fully mixed songs, by mixing and matching the outputs from various preprocessing, input representations, and core beat tracking algorithms. For terminology, I'll say that this library implements some meta-algorithms made from beat tracking algorithms.

A non-goal of the beat-tracking project is to re-implement the beat tracking algorithms from scratch (unlike my [pitch-detection](https://github.com/sevagh/pitch-detection) collection). Every beat tracker used in the project is from an external implementation (madmom, librosa, essentia, etc.).

### Consensus

The central idea to this repo is to combine multiple beat tracking results to take a consensus. This is based on the simple principle that the more beat locations that agree (say, across different pre-processing or input representations, or different beat tracking algorithms), the stronger the evidence that this might be a beat. Note that I have no evidence that this principle is true or makes sense - it's just an experiment.

In all of the meta-algorithms, the next-to-last step returns a sorted numpy ndarray containing every predicted beat location in seconds:
```
all_beats = [0.435356 0.439430 0.959355 0.960317 ...]
```

To compute a consensus of beats from this array, there are three arguments.
1. `--beat-nearness-threshold <float, seconds>` which specifies how close in seconds beats can be near each other to be considered the same beat.
2. `--consensus-ratio <float ratio>` specifies a ratio of how many beat locations (out of maximum possible agreement) need to agree for the event to be considered a beat.
3. `--beat-selection-strategy mean|median|first` decides how to transform a list of similar beat locations (within the nearness threshold of each other) into a list of final beat locations over which to overlay clicks.


From the above example, this is what consensus could look like with a threshold of 0.05 (50ms) and a consensus ratio of 0.66, so two out of three beat tracking algorithms:
```
beats1 = beat_tracker1.beats(x)
beats2 = beat_tracker2.beats(x)
beats3 = beat_tracker3.beats(x)
...

# concat all sorted beat results
all_beats = [0.435 0.439 0.959 0.960 1.216 1.273]

# group those within 0.05 of each other
thresholded_beats = [[0.435 0.439] [0.959 0.960] [1.216] [1.275]]

# count them
beat_counts = [2 2 1 1]

# only the first two beats have >= 2/3 agreements
# use first beat from the collection to represent the group
final_beats = [0.435 0.959]
```

### Table of algorithms and meta-algorithms

Table of algorithms (the number is used in the command-line options e.g. `beat-tracking --algo 1`):

| Number | Name | Reference |
|--------|------|-------|
| 1      | BTrack | https://github.com/adamstark/BTrack |

Meta-algorithms (sometimes inspired by other papers or methods, built from combinations of preprocessing and beat-tracking algorithms).

If multiple algorithms (comma-separated) are specified, **all meta-algorithm automatically incorporates a consensus across all specified algorithms.**

| Number | Name | Description |
|--------|------|-------------|
| 1      | Basic  | Applies a single or multi-algorithm consensus directly to the mixed song |
| 2      | Percussive  | Apply a single or multi-algorithm consensus to the percussive output of median-filtering harmonic-percussive source separation |
| 3      | Kick-snare percussive consensus | Apply a single or multi-algorithm consensus to the percussion separation of the two frequency bands that contain the kick and snare, i.e. 0-120Hz and 200-500Hz. Concept loosely based on the paper [Percussive Beat tracking using real-time median filtering](http://www.adamstark.co.uk/pdf/papers/percussive-beat-tracking-2013.pdf) |

### Inputs and outputs

The input of beat-tracking is always a single wav file containing the music or audio clip in which you want to track beats.

The primary output of beat-tracking is a wav file containing the input with added clicks or beeps on the beat positions. The listener should then decide if the clicks align with their own foot tapping instinct.

Graphical outputs: by using the option `--debug-plots`, additional plots will be displayed that annotate the beat locations on both the waveform and spectrogram. If applicable, various intermediate stages of the chosen meta-algorithm will be displayed as well.

### Meta-algorithm 1: regular single beat tracker

This just applies a single beat tracker directly on the fully mixed song.

Usage:

```python
$ beat-tracking --meta-algo 1 --algo dbnb input.wav output.wav
```

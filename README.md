# headbang.py

headbang.py is a single-file beat-tracking meta-algorithm for accurate beat tracking in fully mixed progressive metal songs.

The input is a wav file containing the fully mixed song you want to track beats for. The inputs tested were mainly from modern prog metal/djent bands (e.g. Periphery, Tesseract, Volumes, Mestis, Vitalism, Anup Sastry, Intervals), acquired with youtube-dl. I found that the algorithm works rather well for many songs, even non-instrumental tracks with vocals.

The output is the same input wav file with clicks on the predicted beat locations, and a txt file containing beat locations (timestamps, in seconds). The beat location file could be used by other tools e.g. to generate some animation synchronized to a song. The user should expect the clicks to line up with their own foot tapping (or headbanging) instincts, even during challenging segments of the song e.g. syncopation, tempo or time-signature transitions, etc.

The result is achieved by considering a consensus or ensemble of the outputs from various different beat and onset detection algorithms applied to the input song (with and without preprocessing). A non-goal of headbang.py is the implementation of any beat tracking or onset detection algorithms from scratch. All of the core algorithms used in the project are from external sources.

### Algorithm details

The algorithm is described in more detail, and with helpful visuals, on the project's github-pages site: https://sevagh.github.io/headbang.py

## Installation and usage

headbang.py has been written and verified with Python 3.8 on AMD64 machines running Fedora 32 Linux. However, there shouldn't be any problems running it on different machines if the requirements can be successfully installed.

The only sticking point is that the [BTrack](https://github.com/adamstark/BTrack) package is not on pip, and needs to be installed manually. My [fork](https://github.com/sevagh/BTrack) supports a Python 3.8 install:

```
sevagh:~ $ git clone https://github.com/sevagh/BTrack
sevagh:~ $ cd BTrack/modules-and-plug-ins/python-module
sevagh:python-module $ python3.8 setup.py build

# install to your system
sevagh:python-module $ sudo python3.8 setup.py install

# install for your local user
sevagh:python-module $ pip3.8 install --user -e .
```

You can choose to omit BTrack (and run the program without `--algorithm 8`), but I find that it's one of the useful members of the consensus for percussive metal songs. The full list of 8 beat trackers are:
* 4 from [madmom.features.beats](https://madmom.readthedocs.io/en/latest/modules/features/beats.html): RNNBeatProcessor activation -> DBNBeatTrackingProcessor, BeatTrackingProcessor, CRFBeatDetectionProcessor, BeatDetectionProcessor
* 2 from Essentia: [BeatTrackerMultiFeature](https://essentia.upf.edu/reference/std_BeatTrackerMultiFeature.html) and [BeatTrackerDegara](https://essentia.upf.edu/reference/std_BeatTrackerDegara.html)
* 1 from librosa: [librosa.beat.beat_track](https://librosa.org/doc/latest/generated/librosa.beat.beat_track.html)
* [BTrack](https://github.com/adamstark/BTrack), mentioned above

More beat trackers are always welcome. The results would probably get better if more diverse beat trackers are added to the group.

There are many arguments, but for a prog metal song, you should be fine with the defaults. Of course, I always encourage experimentation. The [github-pages site](https://sevagh.github.io/headbang.py) should provide clarity on which part of the algorithm is affected by the various input arguments.

```
sevagh:headbang.py $ pip install --user -r ./requirements.txt
sevagh:headbang.py $ ./headbang.py --help
[   INFO   ] MusicExtractorSVM: no classifier models were configured by default
usage: headbang.py [-h] [--algorithms ALGORITHMS]
                   [--beat-near-threshold BEAT_NEAR_THRESHOLD]
                   [--consensus-ratio CONSENSUS_RATIO]
                   [--max-no-beats MAX_NO_BEATS]
                   [--onset-near-threshold ONSET_NEAR_THRESHOLD]
                   [--onset-silence-threshold ONSET_SILENCE_THRESHOLD]
                   [--harmonic-margin HARMONIC_MARGIN]
                   [--harmonic-frame HARMONIC_FRAME]
                   [--percussive-margin PERCUSSIVE_MARGIN]
                   [--percussive-frame PERCUSSIVE_FRAME]
                   [--fast-attack-ms FAST_ATTACK_MS]
                   [--slow-attack-ms SLOW_ATTACK_MS]
                   [--release-ms RELEASE_MS]
                   [--power-memory-ms POWER_MEMORY_MS] [--n-pool N_POOL]
                   [--show-plots]
                   wav_in wav_out

Accurate percussive beat tracking for metal songs

positional arguments:
  wav_in                input wav file
  wav_out               output wav file

optional arguments:
  -h, --help            show this help message and exit
  --n-pool N_POOL       How many threads to use in multiprocessing pool
                        (default: 7)
  --show-plots          Display plots of intermediate steps describing the
                        algorithm using matplotlib (default: False)

beat arguments:
  --algorithms ALGORITHMS
                        List of beat tracking algorithms to apply. Btrack
                        omitted by default (default: 1,2,3,4,5,6,7)
  --beat-near-threshold BEAT_NEAR_THRESHOLD
                        How close beats should be in seconds to be considered
                        the same beat (default: 0.1)
  --consensus-ratio CONSENSUS_RATIO
                        How many (out of the maximum possible) beat locations
                        should agree (default: 0.5)

onsets arguments:
  --max-no-beats MAX_NO_BEATS
                        Segments with missing beats to substitute onsets
                        (default: 2.0)
  --onset-near-threshold ONSET_NEAR_THRESHOLD
                        How close onsets should be in seconds when
                        supplementing onset information (default: 0.1)
  --onset-silence-threshold ONSET_SILENCE_THRESHOLD
                        Silence threshold (default: 0.035)

hpss arguments:
  --harmonic-margin HARMONIC_MARGIN
                        Separation margin for HPSS harmonic iteration
                        (default: 2.0)
  --harmonic-frame HARMONIC_FRAME
                        T-F/frame size for HPSS harmonic iteration (default:
                        4096)
  --percussive-margin PERCUSSIVE_MARGIN
                        Separation margin for HPSS percussive iteration
                        (default: 2.0)
  --percussive-frame PERCUSSIVE_FRAME
                        T-F/frame size for HPSS percussive iteration
                        (default: 256)

transient shaper arguments:
  --fast-attack-ms FAST_ATTACK_MS
                        Fast attack (ms) (default: 1)
  --slow-attack-ms SLOW_ATTACK_MS
                        Slow attack (ms) (default: 15)
  --release-ms RELEASE_MS
                        Release (ms) (default: 20)
  --power-memory-ms POWER_MEMORY_MS
                        Power filter memory (ms) (default: 1)
```

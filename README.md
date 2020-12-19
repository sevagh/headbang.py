# headbang.py

headbang.py is a single-file beat-tracking meta-algorithm for accurate beat tracking in fully mixed progressive metal songs.

The input is a wav file containing the fully mixed song you want to track beats for. The inputs tested were mainly from modern prog metal/djent bands (e.g. Periphery, Tesseract, Volumes, Mestis, Vitalism, Anup Sastry, Intervals), acquired with youtube-dl. I found that the algorithm works rather well for many songs, even non-instrumental tracks with vocals.

The output is the same input wav file with clicks on the predicted beat locations, and a txt file containing beat locations (timestamps, in seconds). The beat location file could be used by other tools e.g. to generate some animation synchronized to a song. The user should expect the clicks to line up with their own foot tapping (or headbanging) instincts, even during challenging segments of the song e.g. syncopation, tempo or time-signature transitions, etc.

The result is achieved by considering a consensus or ensemble of the outputs from various different beat and onset detection algorithms applied to the input song (with and without preprocessing). A non-goal of headbang.py is the implementation of any beat tracking or onset detection algorithms from scratch. All of the core algorithms used in the project are from external sources.

## Algorithm details

The algorithm is described in more detail, and with helpful visuals, on the project's github-pages site: https://sevagh.github.io/headbang.py

In a nutshell:
1. Apply a mix of beat trackers on the input song with no preprocessing, specified by `--algorithms 1,2,3,...`
2. Group beats that are near each other within a configurable time interval, specified by `--beat-near-threshold`
3. Count beats in each group. The first beat in each group is the final beat selected
4. Select the first beat from groups that achieved consensus, configured by `--consensus-ratio` (e.g. if 0.5, then at least half of the algorithms must agree on a beat)
5. Create a "percussive-attack-enhanced" signal by applying percussive source separation followed by transient enhancing to emphasize percussive attacks from the input signal
6. Apply onset detection on the percussive-attack-enhanced signal, weighted towards percussion onset detection
7. Align the final consensus beat locations with the percussive-attack-enhanced onset locations. This should be a better job of weeding out strange beats that don't hit at the same time as the drums
8. Finally, if any long passages of the song have no beats, supplement these with the percussive-attack-enhanced onsets

## Usage

headbang.py has been written and verified with Python 3.8 on AMD64 machines running Fedora 32 Linux. However, there shouldn't be any problems running it if the requirements can be successfully installed.

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

## headbang.py

headbang.py is a single-file meta-algorithm for accurate percussive beat tracking in fully mixed progressive metal songs. It considers the consensus of multiple diverse beat and onset detectors.

The goal is to only predict beats that align with strong percussive onsets. For example, during a segment of the song where the drums are silent, there may be a silent/implicit beat, but headbang.py will not emit any predictions.

Instructions for how to install the dependencies and run headbang.py are in [the source code's README](https://github.com/sevagh/headbang.py). Feel free to post any questions, concerns, or contributions via GitHub issues.

### Examples

I can't include full songs due to copyright issues. Here are 3 excerpts of good outputs from headbang.py (using the default settings) on some songs:
* [Periphery - The Bad Thing](./example_bad_thing.wav) - accurate tracking during accented off-beats 
* [Periphery - Omega](./example_omega.wav) - accurate clicks across a time signature/tempo transition
* [Kadinja - GLHF](./example_glhf.wav) - stable clicks throughout an action-packed section

### Algorithm

#### Input

The input is provided as a single audio file containing the mixed song. This can be a full length typical metal song:

![input_waveform](./input_waveform.png)

Note that the plots were generated with a small segment (10s) extracted from a full song to demonstrate the algorithm more clearly. For best beat results, it's better to pass in the full song. I've tested a range of 3-16-minute songs with accurate tracking throughout. A small, isolated segment of a song will generally lead to bad beat tracking results.

#### Multiple beat tracking algorithms

8 beat trackers are applied on the input directly (without preprocessing):

![input_waveform_all_beats](./input_waveform_all_beats.png)

The list of beat trackers is:
1. [madmom](https://madmom.readthedocs.io/en/latest/modules/features/beats.html) RNNBeatProcessor -> DBNBeatTrackingProcessor
2. madmom RNNBeatProcessor -> BeatTrackingProcessor
3. madmom RNNBeatProcessor -> CRFBeatDetectionProcessor
4. madmom RNNBeatProcessor -> BeatDetectionProcessor
5. [Essentia BeatTrackerMultiFeature](https://essentia.upf.edu/reference/std_BeatTrackerMultiFeature.html)
6. [Essentia BeatTrackerDegara](https://essentia.upf.edu/reference/std_BeatTrackerDegara.html)
7. [librosa beat_track](https://librosa.org/doc/latest/generated/librosa.beat.beat_track.html)
8. [BTrack](https://github.com/adamstark/BTrack)

#### "True beats" are chosen by consensus

A true beat is marked when 4 out of the 8 beat trackers agree (i.e. `--consensus-ratio=0.5`x`len(algorithms)` predict a beat within the same `--beat-near-threshold`, by default 100ms):

![input_waveform_beat_consensus](input_waveform_beat_consensus.png)

#### Percussive onsets

Separately, the percussive component of the input signal is separated with [median-filtering](https://librosa.org/doc/0.8.0/generated/librosa.decompose.hpss.html):

![percussive_hpss](percussive_hpss.png)

Then, I apply my [multi-band transient enhancer](https://gitlab.com/sevagh/multiband-transient-shaper), which is an adaptation of the SPL differential envelope transient shaper, to enhance the percussive attacks and gate the sustained frequencies:

![percussive_transient_enhanced](percussive_transient_enhanced.png)

Onset detection is performed using a combination of 3 onset detection functions, hfc, flux, and rms (from [Essentia](https://essentia.upf.edu/reference/streaming_OnsetDetection.html)), weighted most heavily on hfc for percussive event detection:

![percussive_onsets](percussive_onsets.png)

#### Aligning beats with percussive onsets

The second-last step of the algorithm is to align the consensus beats with the percussive onset events discovered in the previous step.

This eliminates predicted beats that don't fall on a percussive attack:

![input_waveform_beats_onsets_aligned](input_waveform_beats_onsets_aligned.png)

#### Final beats

The final step verifies that there are any long gaps with no beats (`--max-no-beats`, 2 seconds by default). If there are gaps, percussive onsets are inserted. It's assumed that the beat trackers "got confused" and that it's better to fall back to strong percussive attacks to maintain a click continuity.

If the song truly contains no drum hits during that period, then there are no percussive onsets either, so headbang.py maintains the necessary silence.

The final waveform contains consensus beats supplemented with extra percussive onsets:

![input_waveform_final_beats](input_waveform_final_beats.png)

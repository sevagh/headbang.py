# headbang.py

headbang.py is a beat-tracking meta-algorithm for accurate beat tracking in fully mixed progressive metal songs. It's part of my music information retrieval project for MUMT 621.

The input is a wav file containing the fully mixed song you want to track beats for. The inputs tested were mainly from modern prog metal/djent bands (e.g. Periphery, Tesseract, Volumes, Mestis, Vitalism, Anup Sastry, Intervals), acquired with youtube-dl. I found that the algorithm works rather well for many songs, even non-instrumental tracks with vocals.

The output is the same input wav file with clicks on the predicted beat locations, and a txt file containing beat locations (timestamps, in seconds). The beat location file could be used by other tools e.g. to generate some animation synchronized to a song. The user should expect the clicks to line up with their own foot tapping (or headbanging) instincts, even during challenging segments of the song e.g. syncopation, tempo or time-signature transitions, etc.

The result is achieved by considering a consensus or ensemble of the outputs from various different beat and onset detection algorithms applied to the input song (with and without preprocessing). A non-goal of headbang.py is the implementation of any beat tracking or onset detection algorithms from scratch. All of the core algorithms used in the project are from external sources.

### Algorithm details

The algorithm is described in more detail, and with helpful visuals, on the project's github-pages site: https://sevagh.github.io/headbang.py

**nb** For a strange reason, I get significantly worse outputs from my laptop compared to my desktop. I'll need to investigate more. I suspect one of the beat tracking algorithms runs itself in a "low accuracy low performance" mode. If you feel that you're getting totally inaccurate results, let me know by opening an issue.

## Installation and usage

headbang.py has been written and verified with Python 3.8 on AMD64 machines running Fedora 32 Linux. However, there shouldn't be any problems running it on different machines if the requirements can be successfully installed.

The only sticking point is that the [BTrack](https://github.com/adamstark/BTrack) package is not on pip, and needs to be installed manually. My [fork](https://github.com/sevagh/BTrack) supports a Python 3.8 install:

```
sevagh:~ $ git clone https://github.com/sevagh/BTrack
sevagh:~ $ cd BTrack/modules-and-plug-ins/python-module
sevagh:python-module $ python3.8 setup.py build

# install to your system
sevagh:python-module $ sudo python3.8 setup.py install

# OR #

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

# headbang-dashboard

## Installation and usage

Installing groove-dashboard is more complicated than `pip install -r requirements.txt`, due to using two libraries that are not on PyPi and need to be built from source:
* https://github.com/CMU-Perceptual-Computing-Lab/openpose
* https://github.com/adamstark/BTrack

I followed both of their instructions for building and installing the Python module successfully on Python 3.7, Fedora 32, amd64.

groove-dashboard is opinionated and doesn't have many arguments or flags. Call it on the input video file (any format supported by [moviepy](https://github.com/Zulko/moviepy)):
```
$ groove-dashboard input.mp4 output.mp4 --wav-click-prefix output
[   INFO   ] MusicExtractorSVM: no classifier models were configured by default
Starting OpenPose Python Wrapper...
Auto-detecting all available GPUs... Detected 1 GPU(s), using 1 of them starting at GPU 0.
100%|████████████████████████████████████████████████████████████████████████████████████████| 845/845 [06:16<00:00,  2.24it/s]
Getting beat locations using consensus beat tracking
Marking beat and head bop positions on output frames
845it [00:00, 4221.41it/s]
Writing output mp4 groove dashboard
Moviepy - Building video output.mp4.
MoviePy - Writing audio in outputTEMP_MPY_wvf_snd.mp3
MoviePy - Done.
Moviepy - Writing video output.mp4

Moviepy - Done !
Moviepy - video ready output.mp4
Writing clicks overlaid on groove positions to wav file
MoviePy - Writing audio in output-beat.wav
MoviePy - Done.
MoviePy - Writing audio in output-bop.wav
MoviePy - Done.
MoviePy - Writing audio in output-groove.wav
MoviePy - Done.
```
The output mp4 video contains the original video and audio, overlaid with plots and annotations created by groove-dashboard. Additionally, if `--wav-click-prefix` is set, 3 wav clips are also output, overlaying clicks on beat, headbang, and "groove" (i.e. synchronous beat and headbang) locations - click tracks are the best way to observe event detection in MIR (in my opinion).



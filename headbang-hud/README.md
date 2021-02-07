# headbang-dashboard

headbang-dashboard is a joint music information retrieval (MIR) and computer vision project, presented as my final project for MUMT 621.

## Overview

The goal of headbang-dashboard to analyze videos that contain metal music (e.g. metal concerts or guitar covers) and at least one human subject (musician, audience, or otherwise) who is headbanging (moving their head and neck vertically and vigorously on the beat). The dashboard will track and plot peaks of head motion alongside different signal-based predictors of "groove" and the outputs of MIR algorithms e.g. beat tracking.

The hypothesis is that certain parts of metal songs are so groovy that they impel either the musician or the audience (or both) to headbang on the beat. If these moments can be identified and displayed alongside MIR metrics, they could give us insight into which musical properties induce headbanging.

Detail:
insert static image here describing each annotation

insert gifs here

### It works best on...

What sort of videos should you expect good results from?

The two types of video I tested are:
* Footage of solo guitar players playing a metal song or riff and headbanging along to it
* Metal concert footage showing the band or audience headbanging

Solo guitar footage presents an MIR challenge due to the necessity of percussion to create reliable beat tracking results. There should ideally be a drum backing track present so that the audio-based beat tracking and percussive audio metrics are clear.

Metal concert footage presents a computer vision challenge; the videos are often dark with light shows, contains huge crowds of people in motion, change shot frequently and are shaky. Also, music recorded from metal concerts has a lot of audience noise (screaming, etc.) and potentially muffled music, making it challenging for MIR as well.


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

## Groove MIR features

The idea for groove-dashboard started from this paper[[1]](#1), which associates audio signal features or MIR features to human judgements of groove. The paper defines groove as follows:

>The experience of groove is associated with the urge to move to a musical rhythm

From this definition it's natural to look towards the field of computer vision and pose estimation to track the motion jointly with musical measures of groove.

The paper describes several signal-based metrics or MIR features that have varying correlation with the groove judgements of human test subjects, many of which are available in the [MIR toolbox](https://www.jyu.fi/hytk/fi/laitokset/mutku/en/research/materials/mirtoolbox)[[5]](#5). Those selected for use in groove-dashboard are marked with :heavy_check_mark:, those omitted for being uncorrelated are marked with :x:, and those that are impossible to implement (missing information, difficulty) are marked with :large_orange_diamond:.

| Measure | Included | Description |
|---------|----------|-------------|
| Spectral flux in the 0-50Hz, 50-100Hz, and 100-200Hz bands | :heavy_check_mark: | Spectral flux is the measure of the variability of the spectrum of a signal over time. Spectral flux is associated with rhythm and the desire to move[[2]](#2), and is also used in onset detection[[3]](#3). The lower frequency bands 0-200Hz contain the bass component, which is in turn associated with driving the motive to dance or groove. In groove-dashboard, this is implemented by lowpass filtering the music of the video being analyzed in the 0-200Hz range with [scipy.signal.butter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html), and feeding it to the [librosa onset_strength](https://librosa.org/doc/main/generated/librosa.onset.onset_strength.html#librosa.onset.onset_strength) and [onset_detect](https://librosa.org/doc/main/generated/librosa.onset.onset_detect.html#librosa.onset.onset_detect) functions (which use spectral flux[[4]](#4)). |
| Attack time | :heavy_check_mark: | Attack in music refers to how long it takes after the note onset  (i.e. start of the note) for the note to reach full energy. Musicians use short notes to induce groove[[9]](#9), meaning that fast attack times are correlated with groove. It is implemented using the [Essentia Envelope](https://essentia.upf.edu/reference/std_Envelope.html) and [LogAttackTime](https://essentia.upf.edu/reference/std_LogAttackTime.html) algorithms. |
| RMS standard deviation | :heavy_check_mark: | The standard deviation in the RMS (root mean square) of the signal as a measure of _variability_ in the signal, which positively correlates with groove. It is implemented using [librosa rms](https://librosa.org/doc/main/generated/librosa.feature.rms.html) and [numpy std](https://numpy.org/doc/stable/reference/generated/numpy.std.html). |
| Pulse clarity | :x: | Pulse clarity is "considered as a high-level musical dimension that conveys how easily in a given musical piece, or a particular moment during that piece, listeners can perceive the underlying rhythmic or metrical pulsation"[[6]](#6). It was found to have low correlation with groove ratings. |
| Event density | :x: | Event density[[5]](#5) is a measure of the number of note onsets per second. It was found to have low correlation with groove ratings. |
|  Beat salience | :x: | Beat salience[[7]](#7) is a measure of "the degree of repetitive rhythmical patterning around comfortable movement rate", found to have a low correlation with groove ratings. |
| Variance event density | :large_orange_diamond: | Variance event density is a new measure described in the groove paper, consisting of the following steps: 1) given the onset curve, event density estimates are obtained by calculating the variance in the onset curve between beats (beat-to-beat variance), and 2) calculating the mean of the beat-to-beat variance estimates. The variance event density calculation requires ground-truth knowledge of beat locations, which is not available to us. |
| Syncopation | :large_orange_diamond: | Syncopation has a strong association with groove[[10]](#10), however it is a complex subject. The [Python Syncopation Toolkit](https://github.com/Music-Cognition-Lab/SynPy3) (paper[[8]](#8)) was explored for ideas but it is not a simple "syncopation tracker" for arbitrary music clips |

## Beat tracking

Strong beats are associated with groove[[11]](#11), [[12]](#12). Despite good scores in MIREX evaluations, in my experience the outputs of out-of-the-box beat trackers were not perceptually accurate in metal songs. I set out to create a more robust "percussion-aligned metal beat tracker", or "HeadbangBeatTracker", which was split out into a separate repo: [headbang.py](https://github.com/sevagh/headbang.py).

An overview of the algorithm, samples, comparisons, and evaluation results are available at [https://sevagh.github.io/headbang.py](https://sevagh.github.io/headbang.py/). In essence, it's a consensus (or ensemble) beat tracker which combines beats from 8 different beat tracking algorithms and aligns them with onsets detected from the percussive part of the signal. It was tested thoroughly on fully mixed metal songs and found to produce better results than any beat tracker used alone.

## 2D pose estimation with OpenPose

The pose estimation component of groove-dashboard was inspired by the preprinted paper [[14]](#14), at a high level. That paper analyzes beat synchrony of salsa dancers' foot motion. The ideas borrowed were to take the 2D pose detected keypoints of the body parts of interest (in their case feet, in my case head/neck) from OpenPose, normalize the coordinates, record the normalized coordinates per frame of the video, and use peak finding to estimate peaks in motion (for them foot strikes, for me headbangs).

[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) is a "real-time multi-person keypoint detection library for body, face, hands, and foot estimation"[[13]](#13), written in C++, along with a Python wrapper library.

The base of the pose estimation module was taken from the [Tryolabs Norfair demo](https://github.com/tryolabs/norfair/tree/master/demos/openpose). Norfair is a Python library that takes the raw pose estimation outputs of the underlying pose detector (in this case, OpenPose) and provides a higher-level API for drawing tracked objects. In groove-dashboard, Norfair is used simply to draw the pose estimation of the human subject in the output video, to indicate exactly what OpenPose was detecting.

The actual calculation of head motion and peaks are done from the raw OpenPose outputs. OpenPose is configured to use the BODY_25 model, which is their fastest performing pose detector. The keypoints `[0, 1, 15, 16, 17, 18]` correspond to the nose, neck, right eye, left eye, right ear, and left ear respectively.

When a frame of a video with a human subject in it is passed to OpenPose for processing, it returns an array of detected 2D (x, y) coordinate positions per keypoint. This is filtered to only give us the head and neck keypoints mentioned, and then the median y coordinate is taken of the sum (the x coordinate is discarded, since side-to-side head movement is less typical than up-and-down when headbanging or rocking one's head to a beat). The y coordinate is normalized by the size of the video to produce values in the range of [0, 1.0].

In this way, for every frame of the video, we have the median normalized y coordinate of the head and neck region, which should correlate to the torso and head/headbanging motion of the subject. From the cumulative y coordinates of all the frames, [scipy find_peaks_cwt](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks_cwt.html) is used with a width of 10 (e.g. find 1 peak every 10 frames), which gives us robust results for tracking the extreme ranges of motion and not spurious peaks. The parameters were chosen with trial and error.

These peaks in y coordinate motion of the head and torso are called "bops", or the pose/motion analog of a beat. A bop is when the human subject in the video reaches the peak of their headbang - the final goal is to see how often beats and bops line up, which indicates the motion is exactly on the beat. Finally, the tempo of head motion can be estimated by using the median inter-bop interval of the detected bops (or peaks in head motion).

The results can be seen here:

![norfair_bpm](.github/headbop.png)

Observe the Norfair drawn keypoints on my (blurry) head, the *BOP* marker (indicating this frame contains an extreme range of motion, or peak y coordinate), and the tempo estimated from the inter-bop interval. The motion tempo is 81 (stably throughout the test video), which is close to the correct value (I recorded myself headbanging to a metronome set to 80bpm).

![ycoordpeaks](.github/ycoordpeaks.png)

Observe the y coordinate peaks with magenta lines marking the output of the peak picking algorithm, demonstrating the robustness and accuracy of tracking the regular motion.

## Input, code architecture, and outputs

The input is an mp4 file, and the core of groove-dashboard is a simple loop over the frames of the video with OpenCV in [main.py](groove_dashboard/main.py).

diagram is best here

## Discussion, conclusion, etc.

mention this also, another OpenPose music paper - replace beat tracking with pose detection - waw: https://program.ismir2020.net/poster_3-10.html

## References

<a id="1">[1]</a> 
Stupacher, Jan & Hove, Michael & Janata, Petr. (2016). Audio Features Underlying Perceived Groove and Sensorimotor Synchronization in Music. Music Perception. 33. 571-589. 10.1525/mp.2016.33.5.571. 

<a id="2">[2]</a>
Burger, Birgitta & Ahokas, J. Riikka & Keipi, Aaro & Toiviainen, Petri. (2013). Relationships between spectral flux, perceived rhythmic strength, and the propensity to move. 

<a id="3">[3]</a>
Dixon, Simon. (2006). Simple spectrum-based onset detection. 

<a id="4">[4]</a>
Böck, Sebastian, and Gerhard Widmer. “Maximum filter vibrato suppression for onset detection.” 16th International Conference on Digital Audio Effects, Maynooth, Ireland. 2013.

<a id="5">[5]</a>
O. Lartillot and P. Toiviainen, "A Matlab Toolbox for Musical Feature Extraction from Audio," in Proceedings of the 10th International Conference on Digital Audio Effects (DAFx-07) (2007).

<a id="6">[6]</a>
Lartillot, Olivier & Eerola, Tuomas & Toiviainen, Petri & Fornari, Jose. (2008). Multi-Feature Modeling of Pulse Clarity: Design, Validation and Optimization. 521-526. 

<a id="7">[7]</a>
Madison, Guy & Gouyon, Fabien & Ullén, Fredrik & Hörnström, Kalle. (2011). Modeling the Tendency for Music to Induce Movement in Humans: First Correlations With Low-Level Audio Descriptors Across Music Genres. Journal of experimental psychology. Human perception and performance. 37. 1578-94. 10.1037/a0024323. 

<a id="8">[8]</a>
C. Song, M. Pearce, and C. Harte, SynPy: a Python Toolkit for Syncopation Modelling. Maynooth, Ireland, 2015.

<a id="9">[9]</a>
Madison, Guy & Sioros, George. (2014). What musicians do to induce the sensation of groove in simple and complex melodies, and how listeners perceive it. Frontiers in Psychology. 5. 10.3389/fpsyg.2014.00894. 

<a id="10">[10]</a>
Sioros, George & Miron, Marius & Davies, Matthew & Gouyon, Fabien & Madison, Guy. (2014). Syncopation creates the sensation of groove in synthesized music examples. Frontiers in psychology. 5. 1036. 10.3389/fpsyg.2014.01036. 

<a id="11">[11]</a>
Madison G, Gouyon F, Ullen F. Musical groove is correlated with properties of the audio signal as revealed by computational modelling, depending on musical style. In: Proceedings of the SMC 2009—6th Sound and Music Computing Conference. 2009. p. 239–40.

<a id="12">[12]</a>
Madison G, Gouyon F, Ullén F, Hörnström K. Modeling the tendency for music to induce movement in humans: First correlations with low-level audio descriptors across music genres. J Exp Psychol Hum Percept Perform. 2011; 37:1578–1594. pmid:21728462

<a id="13">[13]</a>
Z. Cao, G. Hidalgo, T. Simon, S. -E. Wei and Y. Sheikh, "OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 43, no. 1, pp. 172-186, 1 Jan. 2021, doi: 10.1109/TPAMI.2019.2929257.

<a id="14">[14]</a>
Quantifying music-dance synchrony with the application of a deep learning-based 2D pose estimator
Filip Potempski, Andrea Sabo, Kara K Patterson
bioRxiv 2020.10.09.333617; doi: https://doi.org/10.1101/2020.10.09.333617 

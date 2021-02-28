import numpy
import time
import cv2
import sys
import itertools
import argparse
import sys
import librosa
import gc
import os
import multiprocessing
from moviepy.editor import *
from moviepy.audio.AudioClip import AudioArrayClip
from tempfile import gettempdir
from headbang.params import DEFAULTS
from headbang import HeadbangBeatTracker
from headbang.util import load_wav, overlay_clicks
from headbang.consensus import algo_names
from madmom.io.audio import write_wave_file


def main():
    parser = argparse.ArgumentParser(
        description="Vizualize the headbang beat tracker"
    )

    parser.add_argument("wav_in", type=str, help="wav file to process")
    parser.add_argument("mp4_out", type=str, help="mp4 output path")

    args = parser.parse_args()

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    hbt = HeadbangBeatTracker(pool)

    audio, _ = librosa.load(args.wav_in, sr=44100, dtype=numpy.float32, mono=True)

    # get beat locations
    print("Getting beat locations using consensus beat tracking")
    strong_beat_locations = hbt.beats(audio)

    # get the inner multi-beat-tracker list from headbangbeattracker's consensusbeattracker object
    individual_tracker_beat_locations = hbt.cbt.beat_results

    # blue, yellow, magenta, violet, orange, brown, white
    colors = itertools.cycle([(0, 165, 255), (255, 0, 255), (255, 255, 0), (255, 69, 0), (0, 255, 255), (165, 42, 42), (255, 255, 255)])

    beat_trackers = {
            'headbang': {
                'beats': strong_beat_locations,
                'color': (255, 0, 0), # red
            },
            'consensus': {
                'beats': hbt.beat_consensus,
                'color': (0, 255, 0), # lime green
            },
    }
    for i, algo_name in enumerate(algo_names[1:]):
        beat_trackers[algo_name] = {
            'beats': individual_tracker_beat_locations[i],
            'color': next(colors)
        }

    beat_trackers['onsets'] = {
            'beats': hbt.onsets,
            'color': next(colors)
    }

    for name, bt in beat_trackers.items():
        print('{0}: {1}'.format(name, bt))

    fps = 30

    video_width = 1920
    video_height = 1080

    frame_duration = 1 / fps
    frame_duration_ms = frame_duration * 1000

    total_duration = float(audio.shape[0])/44100.0
    total_frames = total_duration/frame_duration

    total_duration = frame_duration * total_frames

    blank_frame = numpy.zeros((video_height, video_width, 3), numpy.uint8)

    def render_animations(*args, **kwargs):
        video_frame = blank_frame.copy()

        # draw stick figures with text

        # draw some text, names of algorithms etc.
        cv2.putText(
            video_frame,
            "BEAT",
            all_beat_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            all_beat_color,
            3,
            cv2.LINE_AA,
        )
        cv2.line(image, (20,10), (100,10), (255,0,0), 2)


        # adjust color on frames
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        return video_frame

    print("Processing video - rendering animations")
    out_clip = VideoClip(make_frame=render_animations, duration=total_duration)

    audio_clip = AudioFileClip(args.wav_in)
    new_audioclip = CompositeAudioClip([audio_clip])

    out_clip.audio = new_audioclip
    out_clip.write_videofile(args.mp4_out, fps=fps)

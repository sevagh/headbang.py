import numpy
import cv2
import itertools
import argparse
import pandas as pd
import multiprocessing
from moviepy.editor import AudioFileClip, VideoClip, CompositeAudioClip
from headbang import HeadbangBeatTracker
from headbang.util import load_wav


def find_closest(A, target):
    idx = A.searchsorted(target)
    idx = numpy.clip(idx, 1, len(A) - 1)
    left = A[idx - 1]
    right = A[idx]
    idx -= target - left < right - target
    return idx


def main():
    parser = argparse.ArgumentParser(description="Vizualize the headbang beat tracker")

    parser.add_argument("wav_in", type=str, help="wav file to process")
    parser.add_argument("mp4_out", type=str, help="mp4 output path")

    args = parser.parse_args()

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    hbt = HeadbangBeatTracker(pool)

    audio = load_wav(args.wav_in)

    # get beat locations
    print("Getting beat locations using consensus beat tracking")
    strong_beat_locations = hbt.beats(audio)

    # get the inner multi-beat-tracker list from headbangbeattracker's consensusbeattracker object
    individual_tracker_beat_locations = hbt.cbt.beat_results

    colors = [
        (255, 0, 0),  # red
        (0, 255, 0),  # green
        (0, 165, 255),  # blue
        (255, 0, 255),  # magenta
        (255, 255, 0),  # yellow
        (255, 69, 0),  # orange
        (0, 255, 255),  # cyan
        (145, 112, 235),  # blue-violet
    ]

    fps = 30

    video_width = 1920
    video_height = 1080

    frame_duration = 1 / fps

    total_duration = numpy.floor(float(audio.shape[0]) / 44100.0)
    total_frames = int(numpy.ceil(total_duration / frame_duration))

    times_vector = numpy.arange(0, total_duration, frame_duration)

    all_beat_times = individual_tracker_beat_locations + [
        strong_beat_locations,
        hbt.beat_consensus,
    ]

    all_beat_frames = [
        numpy.concatenate(
            (
                numpy.zeros(
                    1,
                ),
                find_closest(times_vector, beat_times),
                numpy.ones(
                    1,
                )
                * (total_frames - 1),
            )
        ).astype(numpy.int)
        for beat_times in all_beat_times
    ]

    off_beat_frames = [
        ((x[1:] + x[:-1]) / 2).astype(numpy.int) for x in all_beat_frames
    ]

    all_positions = []  # []
    for i in range(len(all_beat_frames)):
        x = (
            numpy.empty(
                total_frames,
            )
            * numpy.nan
        )

        x[all_beat_frames[i]] = 1
        x[off_beat_frames[i]] = -1
        a = pd.Series(x)
        all_positions.append(a.interpolate().to_numpy())

    blank_frame = numpy.zeros((video_height, video_width, 3), numpy.uint8)

    box_width = int(video_width / 4)
    box_edges_horiz = numpy.arange(0, video_width + 1, box_width)
    box_centers_horiz = box_edges_horiz[:-1] + int(box_width / 2)

    box_height = int(video_height / 2)
    box_edges_vert = numpy.arange(0, video_height + 1, box_height)
    box_centers_vert = box_edges_vert[:-1] + int(box_height / 2)

    positions = list(itertools.product(box_centers_horiz, box_centers_vert))

    frame_index = 0

    def render_animations(*args, **kwargs):
        nonlocal frame_index
        video_frame = blank_frame.copy()

        for i, beats in enumerate(all_beat_frames):
            center = positions[i]
            try:
                interpolated_pos = all_positions[i][frame_index]
            except IndexError:
                interpolated_pos = 0

            current_position = (
                center[0],
                int(center[1] + (box_height / 2 - 100) * interpolated_pos),
            )

            # draw some text, names of algorithms etc.
            cv2.putText(
                video_frame,
                str(i),
                current_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                colors[i],
                3,
                cv2.LINE_AA,
            )

        frame_index += 1
        return video_frame

    print("Processing video - rendering animations")
    out_clip = VideoClip(make_frame=render_animations, duration=total_duration)

    audio_clip = AudioFileClip(args.wav_in)
    new_audioclip = CompositeAudioClip([audio_clip])

    out_clip.audio = new_audioclip
    out_clip.write_videofile(args.mp4_out, fps=fps)

#!/usr/bin/env python

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
import scipy
from scipy.signal import find_peaks_cwt
from tempfile import gettempdir

from headbang import HeadbangBeatTracker

openpose_install_path = "/home/sevagh/thirdparty-repos/openpose"

openpose_dir = openpose_install_path
sys.path.append(openpose_dir + "/build/python/openpose")

import pyopenpose as op


class OpenposeDetector:
    # nose, neck, right eye, left eye, right ear, left ear
    face_neck_keypoints = [0, 1, 15, 16, 17, 18]
    confidence_threshold = 0.2

    def __init__(self, custom_keypoints=None):
        config = {}
        # config["dir"] = openpose_install_path
        config["logging_level"] = 3
        config["net_resolution"] = "320x320"  # 320x176
        # config["output_resolution"] = "-1x768"  # 320x176
        config["model_pose"] = "BODY_25"
        config["alpha_pose"] = 0.6
        config["scale_gap"] = 0.3
        config["scale_number"] = 1
        # config["keypoint_scale"] = 4 # scale to -1,1
        config["render_threshold"] = 0.05
        config[
            "num_gpu_start"
        ] = 0  # If GPU version is built, and multiple GPUs are available, set the ID here
        config["disable_blending"] = False

        config["model_folder"] = openpose_dir + "/models/"
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(config)
        self.opWrapper.start()

        if custom_keypoints:
            keypoints = [int(i) for i in custom_keypoints.split(",")]
            self.keypoints = keypoints
        else:
            self.keypoints = OpenposeDetector.face_neck_keypoints

    def detect_pose(self, image):
        datum = op.Datum()
        datum.cvInputData = image
        ret = self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        if not ret:
            raise ValueError("couldn't emplaceAndPop")
        return datum.poseKeypoints, datum.cvOutputData

    def process_frame(self, frame):
        tracked_objects = None
        detected_poses, outframe = self.detect_pose(frame)

        median_x = None
        median_y = None

        if detected_poses is not None:
            detected_poses = detected_poses[0]
            # array of (x, y) coordinates of the head/neck
            head_poses = [
                (d[0], d[1])
                for i, d in enumerate(detected_poses)
                if i in self.keypoints and d[2] > OpenposeDetector.confidence_threshold
            ]

            if head_poses:
                head_poses = numpy.asarray(head_poses)
                median_coords = numpy.median(head_poses, axis=0)
                median_x = median_coords[0]
                median_y = median_coords[1]

        return median_x, median_y, outframe


def bpm_from_beats(beats):
    if beats.size == 0:
        return 0
    m_res = scipy.stats.linregress(numpy.arange(len(beats)), beats)

    first_beat = m_res.intercept
    beat_step = m_res.slope

    return 60 / beat_step


def main():
    parser = argparse.ArgumentParser(
        description="Track human pose in videos with music alongside groove metrics and beat tracking"
    )
    parser.add_argument(
        "--custom-keypoints", type=str, help="Override the default face/neck keypoints"
    )
    parser.add_argument("mp4_in", type=str, help="mp4 file to process")
    parser.add_argument("mp4_out", type=str, help="mp4 output path")

    args = parser.parse_args()

    pose_tracker = OpenposeDetector(custom_keypoints=args.custom_keypoints)

    video_path = args.mp4_in
    cap = cv2.VideoCapture(video_path)

    pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
    hbt = HeadbangBeatTracker(pool)

    audio, _ = librosa.load(video_path, sr=44100, dtype=numpy.float32, mono=True)

    # get beat locations
    print("Getting beat locations using consensus beat tracking")
    strong_beat_locations = hbt.beats(audio)

    # pre onset alignment
    all_beat_locations = hbt.beat_consensus

    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    frame_duration = 1 / fps
    frame_duration_ms = frame_duration * 1000

    total_duration = frame_duration * total_frames

    chunk_size = int(numpy.round(frame_duration * 44100))

    video_width = 1920
    video_height = 1080

    plot_width = int(video_width / 4)
    plot_height = int(video_height / 4.5)

    bop_pos = (100, int(video_height - 50))
    bop_bpm_pos = (280, int(video_height - 50))
    bop_color = (255, 255, 0)

    all_beat_pos = (635, int(video_height - 50))
    all_beat_bpm_pos = (860, int(video_height - 50))
    all_beat_color = (255, 0, 0)

    groove_pos = (1200, int(video_height - 50))
    groove_color = (0, 255, 0)

    strong_beat_pos = (1600, int(video_height - 50))
    strong_beat_color = (0, 165, 255)

    alpha = 0.90

    pose_frame = None

    blank_frame = numpy.zeros((video_height, video_width, 3), numpy.uint8)
    all_y_coords = []

    def process_first_pass(*args, **kwargs):
        grabbed, frame = cap.read()
        if not grabbed:
            return blank_frame

        frame = cv2.resize(frame, (video_width, video_height))

        # update latest pose frame
        median_x, median_y, out_frame = pose_tracker.process_frame(frame)
        x_norm = 0
        y_norm = 0

        if median_x is not None:
            x_norm = median_x / video_width
        if median_y is not None:
            y_norm = median_y / video_height

        # record normalized y coords per frame to find peaks in motion
        all_y_coords.append(y_norm)

        # draw semi-transparent rectangle
        hud_overlay = out_frame.copy()

        cv2.rectangle(
            hud_overlay,
            (0, video_height - 150),
            (video_width, video_height),
            (0, 0, 0),
            -1,
        )
        video_frame = cv2.addWeighted(hud_overlay, alpha, out_frame, 1 - alpha, 0)

        # adjust color on frames
        video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
        return video_frame

    tmp_mp4 = os.path.join(gettempdir(), "headbang-hud-tmp.mp4")

    print("Processing video - first pass with pose detection")
    out_clip = VideoClip(make_frame=process_first_pass, duration=total_duration)
    out_clip.write_videofile(tmp_mp4, fps=fps)

    # get head bop locations by indexing into time array
    all_time = numpy.linspace(0, frame_duration * total_frames, int(total_frames))

    # use find_peaks_cwt, works well
    bop_locations = all_time[find_peaks_cwt(all_y_coords, numpy.arange(1, 10))]

    event_thresh = 2 * frame_duration

    print("Marking beat and head bop positions on output frames")

    frame_history = 3  # consider this many seconds of history for bpm computation

    all_beats_bpm = 0
    strong_beats_bpm = 0
    bop_bpm = 0

    print("run a gc, just in case...")
    gc.collect()

    # define a function to filter the first video to add more stuff
    def process_second_pass(get_frame_fn, frame_time):
        nonlocal all_beats_bpm, bop_bpm, strong_beats_bpm
        frame = get_frame_fn(frame_time)

        frame_max = frame_time
        frame_min = max(0, frame_time - frame_history)

        all_beat_history = all_beat_locations[
            numpy.where(
                (all_beat_locations >= frame_min) & (all_beat_locations <= frame_max)
            )
        ]

        strong_beat_history = strong_beat_locations[
            numpy.where(
                (strong_beat_locations >= frame_min)
                & (strong_beat_locations <= frame_max)
            )
        ]

        bop_history = bop_locations[
            numpy.where((bop_locations >= frame_min) & (bop_locations <= frame_max))
        ]

        all_beats_bpm_tmp = bpm_from_beats(all_beat_history)
        bop_bpm_tmp = bpm_from_beats(bop_history)

        if not numpy.isnan(all_beats_bpm_tmp):
            all_beats_bpm = all_beats_bpm_tmp

        if not numpy.isnan(bop_bpm_tmp):
            bop_bpm = bop_bpm_tmp

        is_strong_beat = False
        is_beat = False
        is_bop = False
        if any(
            [b for b in all_beat_locations if numpy.abs(b - frame_time) <= event_thresh]
        ):
            is_beat = True
        if any(
            [
                b
                for b in strong_beat_locations
                if numpy.abs(b - frame_time) <= event_thresh
            ]
        ):
            is_strong_beat = True
        if any([b for b in bop_locations if numpy.abs(b - frame_time) <= event_thresh]):
            is_bop = True

        is_groove = is_bop and is_beat

        if is_beat:
            cv2.putText(
                frame,
                "BEAT",
                all_beat_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                all_beat_color,
                3,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            "{0:.2f} bpm".format(all_beats_bpm),
            all_beat_bpm_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            all_beat_color,
            2,
            cv2.LINE_AA,
        )

        if is_strong_beat:
            cv2.putText(
                frame,
                "BEAT+",
                strong_beat_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                strong_beat_color,
                3,
                cv2.LINE_AA,
            )

        if is_bop:
            cv2.putText(
                frame,
                "BOP",
                bop_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                bop_color,
                3,
                cv2.LINE_AA,
            )
        cv2.putText(
            frame,
            "{0:.2f} bpm".format(bop_bpm),
            bop_bpm_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            bop_color,
            2,
            cv2.LINE_AA,
        )

        if is_groove:
            cv2.putText(
                frame,
                "GROOVE",
                groove_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                groove_color,
                3,
                cv2.LINE_AA,
            )
        return frame

    print(
        "Processing video - second pass with marked event locations and estimated tempos"
    )

    print("run a gc, just in case...")
    gc.collect()

    out_clip_tmp = VideoFileClip(tmp_mp4)
    out_clip2 = out_clip_tmp.fl(process_second_pass)

    audio_clip = AudioFileClip(video_path)
    new_audioclip = CompositeAudioClip([audio_clip])

    out_clip2.audio = new_audioclip
    out_clip2.write_videofile(args.mp4_out, fps=fps)

    print("cleaning up tmp mp4")
    os.remove(tmp_mp4)


if __name__ == "__main__":
    main()

import numpy
import cv2
import argparse
import librosa
import gc
import os
import multiprocessing
from moviepy.editor import CompositeAudioClip, AudioFileClip, VideoFileClip, VideoClip
from tempfile import gettempdir
from headbang.motion import OpenposeDetector, bpm_from_beats, align_beats_motion
from headbang.params import DEFAULTS
from headbang import HeadbangBeatTracker
from headbang.util import load_wav, overlay_clicks
from madmom.io.audio import write_wave_file


def main():
    parser = argparse.ArgumentParser(
        description="Track human pose in videos with music alongside groove metrics and beat tracking"
    )
    parser.add_argument(
        "--keypoints",
        type=str,
        default=DEFAULTS["pose_keypoints"],
        help="Override the default face keypoints (default=%(default)s)",
    )
    parser.add_argument(
        "--bpm-history",
        type=float,
        default=DEFAULTS["bpm_history"],
        help="History of video (in seconds) to be included in the window of current bpm computation (default=%(default)s)",
    )
    parser.add_argument(
        "--event-threshold-frames",
        type=int,
        default=DEFAULTS["event_thresh_frames"],
        help="Threshold in number of frames by which an event is considered to be the same (default=%(default)s)",
    )
    parser.add_argument(
        "--debug-motion",
        action="store_true",
        help="Only perform motion detection with matplotlib - no beat tracking",
    )
    parser.add_argument(
        "--experimental-wav-out",
        type=str,
        default="",
        help="wav output path for bop clicks",
    )
    parser.add_argument(
        "--experimental-bop-align",
        type=float,
        default=DEFAULTS["bop_align"],
        help="align bops and beats within this duration window (s) (default=%(default)s)",
    )
    parser.add_argument(
        "--experimental-sick-chain-boundary",
        type=float,
        default=DEFAULTS["sick_chain_boundary"],
        help="time boundary to separate sick portions of a song (default=%(default)s)",
    )
    parser.add_argument(
        "--experimental-sick-chain",
        action="store_true",
        help="identify and display when a sick chain is occuring (according to the boundary argument)",
    )

    parser.add_argument("mp4_in", type=str, help="mp4 file to process")
    parser.add_argument("mp4_out", type=str, help="mp4 output path")

    args = parser.parse_args()

    video_path = args.mp4_in
    cap = cv2.VideoCapture(video_path)

    if args.debug_motion:
        strong_beat_locations = numpy.empty(shape=(1,), dtype=numpy.float32)
        all_beat_locations = numpy.empty(shape=(1,), dtype=numpy.float32)
    else:
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
    frame_duration = 1 / fps

    pose_tracker = OpenposeDetector(
        total_frames,
        frame_duration,
        keypoints=args.keypoints,
    )

    total_duration = frame_duration * total_frames

    video_width = 1920
    video_height = 1080

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

    sick_color = (255, 0, 255)

    alpha = 0.90

    blank_frame = numpy.zeros((video_height, video_width, 3), numpy.uint8)

    def process_first_pass(*args, **kwargs):
        grabbed, frame = cap.read()
        if not grabbed:
            return blank_frame

        frame = cv2.resize(frame, (video_width, video_height))

        # update latest pose frame
        out_frame = pose_tracker.process_frame(frame)

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

    # take top peaks only
    print("Getting peaks of y motion")
    peaks = pose_tracker.find_peaks()
    bop_locations = all_time[peaks]

    aligned_bop_locations = numpy.asarray(
        align_beats_motion(
            all_beat_locations, bop_locations, args.experimental_bop_align
        )
    )

    if args.debug_motion:
        print("Displaying debug y coordinate plot")
        pose_tracker.plot_ycoords()

    event_thresh = args.event_threshold_frames * frame_duration

    print("Marking beat and head bop positions on output frames")

    print("run a gc, just in case...")
    gc.collect()

    all_beats_bpm = 0
    bop_bpm = 0
    time_since_last_groove = None

    # define a function to filter the first video to add more stuff
    def process_second_pass(get_frame_fn, frame_time):
        nonlocal all_beats_bpm, bop_bpm, time_since_last_groove
        frame = get_frame_fn(frame_time)

        frame_max = frame_time
        frame_min = max(0, frame_time - args.bpm_history)

        all_beat_history = all_beat_locations[
            numpy.where(
                (all_beat_locations >= frame_min) & (all_beat_locations <= frame_max)
            )
        ]

        # keep a running history of bops
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
        is_sick = False
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

        if (
            time_since_last_groove is not None
            and frame_time - time_since_last_groove
            <= args.experimental_sick_chain_boundary
        ):
            is_sick = True
        else:
            # sick chain is broken
            time_since_last_groove = None

        if is_groove and time_since_last_groove is None:
            time_since_last_groove = frame_time

        if not args.debug_motion:
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

        if args.experimental_sick_chain and is_sick:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "SICK"

            textsize = cv2.getTextSize(text, font, 26, 12)[0]

            x_text = int((frame.shape[1] - textsize[0]) / 2)
            y_text = int((frame.shape[0] + textsize[1]) / 2) - 75

            cv2.putText(
                frame, text, (x_text, y_text), font, 26, sick_color, 12, cv2.LINE_AA
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

    if args.experimental_wav_out:
        print("Overlaying clicks at bop locations")
        x_stereo = load_wav(video_path, stereo=True)
        x_with_clicks = overlay_clicks(x_stereo, aligned_bop_locations)

        print("Writing output with clicks to {0}".format(args.experimental_wav_out))
        write_wave_file(x_with_clicks, args.experimental_wav_out, sample_rate=44100)

import numpy
import sys
import os
import scipy
from scipy.signal import find_peaks_cwt
import matplotlib.pyplot as plt
from headbang.params import DEFAULTS

openpose_install_path = "/home/sevagh/thirdparty-repos/openpose"

openpose_dir = openpose_install_path
sys.path.append(openpose_dir + "/build/python/openpose")

import pyopenpose as op


class OpenposeDetector:
    undef_coord_default = numpy.nan
    object_limit = 3
    min_confidence = 0.5

    def __init__(
        self,
        n_frames,
        frame_duration,
        keypoints=DEFAULTS["pose_keypoints"],
    ):
        config = {}
        config["logging_level"] = 3
        config["net_resolution"] = "320x320"
        config["model_pose"] = "BODY_25"
        config["alpha_pose"] = 0.6
        config["scale_gap"] = 0.3
        config["scale_number"] = 1
        config["render_threshold"] = 0.05
        config["num_gpu_start"] = 0
        config["disable_blending"] = False

        config["model_folder"] = openpose_dir + "/models/"
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(config)
        self.opWrapper.start()

        self.keypoints = [int(i) for i in keypoints.split(",")]

        self.n_frames = int(n_frames)
        self.all_y_coords = [OpenposeDetector.undef_coord_default] * self.n_frames
        self.frame_idx = 0
        self.frame_duration = frame_duration
        self.total_duration = self.frame_duration * self.n_frames

        print("Started OpenposeDetector for keypoints {0}".format(self.keypoints))

    def detect_pose(self, image):
        datum = op.Datum()
        datum.cvInputData = image
        ret = self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        if not ret:
            raise ValueError("couldn't emplaceAndPop")
        return datum.poseKeypoints, datum.cvOutputData

    def process_frame(self, frame):
        multiple_detected_poses, outframe = self.detect_pose(frame)

        if multiple_detected_poses is not None:
            poses_of_interest = []

            # collect (x, y) coordinates of the head, median across the first object_limit objects
            for detected_poses in multiple_detected_poses[
                : OpenposeDetector.object_limit
            ]:
                for keypoint, d in enumerate(detected_poses):
                    if (
                        keypoint in self.keypoints
                        and d[2] > OpenposeDetector.min_confidence
                    ):
                        poses_of_interest.append((d[0], d[1]))

            poses_of_interest = numpy.asarray(poses_of_interest)
            median_coords = numpy.median(poses_of_interest, axis=0)

            if not numpy.any(numpy.isnan(median_coords)):
                median_y = median_coords[1]
                y_norm = median_y / frame.shape[0]
                self.all_y_coords[self.frame_idx] = y_norm

        self.frame_idx += 1
        return outframe

    def find_peaks(self):
        min_coord = numpy.nanmin(self.all_y_coords)
        adjusted_y_coords = numpy.nan_to_num(self.all_y_coords, nan=min_coord)

        # wavelets are good for peaks
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2631518/
        peaks = find_peaks_cwt(adjusted_y_coords, numpy.arange(2, 4))
        peaks = peaks[numpy.where(numpy.diff(peaks) > 11)[0]]
        return peaks

    def plot_ycoords(self):
        plt.figure(1)
        plt.title("normalized median y coordinate motion")

        plt.xlabel("time (s)")
        plt.ylabel("normalized y coordinate")

        frame_times = numpy.arange(0.0, self.total_duration, self.frame_duration)
        peaks = self.find_peaks()

        y_coords = numpy.asarray(self.all_y_coords)

        plt.plot(
            frame_times,
            y_coords,
            "-D",
            markevery=peaks,
            mec="black",
        )
        plt.grid()
        plt.show()


def bpm_from_beats(beats):
    if beats.size == 0:
        return 0
    m_res = scipy.stats.linregress(numpy.arange(len(beats)), beats)

    first_beat = m_res.intercept
    beat_step = m_res.slope

    return 60 / beat_step


def align_beats_motion(beats, motion, thresh):
    i = 0
    j = 0

    aligned_beats = []
    time_since_last_beat = 0.0

    while i < len(motion) and j < len(beats):
        curr_motion = motion[i]
        curr_beat = beats[j]

        if numpy.abs(curr_motion - curr_beat) <= thresh:
            aligned_beats.append(min(curr_motion, curr_beat))
            i += 1
            j += 1
            continue

        if curr_beat < curr_motion:
            # increment beats
            j += 1
        elif curr_beat > curr_motion:
            i += 1

    return aligned_beats
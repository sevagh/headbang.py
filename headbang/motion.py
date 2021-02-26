import numpy
import sys
import itertools
import os
import scipy
from scipy.signal import find_peaks, find_peaks_cwt, peak_prominences
import matplotlib.pyplot as plt
from defaultlist import defaultlist

openpose_install_path = "/home/sevagh/thirdparty-repos/openpose"

openpose_dir = openpose_install_path
sys.path.append(openpose_dir + "/build/python/openpose")

import pyopenpose as op

marker = itertools.cycle((",", "+", ".", "o", "*"))


class OpenposeDetector:
    # nose, neck, right eye, left eye, right ear, left ear
    face_neck_keypoints = [0, 1, 15, 16, 17, 18]
    confidence_threshold = 0.5
    peak_prominence = 0.2

    def __init__(self, n_frames, custom_keypoints=None):
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

        self.n_frames = int(n_frames)
        self.all_y_coords = [[numpy.nan] * self.n_frames]
        self.frame_idx = 0

    def detect_pose(self, image):
        datum = op.Datum()
        datum.cvInputData = image
        ret = self.opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        if not ret:
            raise ValueError("couldn't emplaceAndPop")
        return datum.poseKeypoints, datum.cvOutputData

    def process_frame(self, frame):
        tracked_objects = None
        multiple_detected_poses, outframe = self.detect_pose(frame)

        median_x = None
        median_y = None

        if multiple_detected_poses is not None:
            # array of (x, y) coordinates of the head/neck
            multiple_poses_of_interest = [
                [
                    (d[0], d[1])
                    for i, d in enumerate(single_detected_poses)
                    if i in self.keypoints
                    and d[2] > OpenposeDetector.confidence_threshold
                ]
                for single_detected_poses in multiple_detected_poses
            ]

            if multiple_poses_of_interest:
                for i, poses_of_interest in enumerate(multiple_poses_of_interest):
                    poses_of_interest = numpy.asarray(poses_of_interest)
                    median_coords = numpy.median(poses_of_interest, axis=0)
                    if not numpy.any(numpy.isnan(median_coords)):
                        median_y = median_coords[1]
                        y_norm = median_y / frame.shape[0]
                        try:
                            self.all_y_coords[i][self.frame_idx] = y_norm
                        except IndexError:
                            self.all_y_coords.append([numpy.nan] * self.n_frames)
                            self.all_y_coords[i][self.frame_idx] = y_norm

        self.frame_idx += 1
        return outframe

    def find_peaks(self):
        peaks = [
            find_peaks(
                numpy.nan_to_num(y_coords), prominence=OpenposeDetector.peak_prominence
            )
            for y_coords in self.all_y_coords
        ]
        peaks = [p[0] for p in peaks]

        prominences = [
            peak_prominences(numpy.nan_to_num(y_coords), peaks[i])
            for i, y_coords in enumerate(self.all_y_coords)
        ]
        prominences = [p[1] for p in prominences]

        strongest_peaks_index = prominences.index(max(prominences, key=sum))

        return peaks, strongest_peaks_index

    def plot_ycoords(self):
        peaks, strongest_peaks_index = self.find_peaks()

        plt.figure(1)
        plt.title("normalized y coordinate motion")

        plt.xlabel("frame")
        plt.ylabel("y coord")

        frames = numpy.arange(self.n_frames)

        for i, y_coords in enumerate(self.all_y_coords):
            y_coords = numpy.asarray(y_coords)
            plt.plot(frames, y_coords, label="obj {0}".format(i))

            if i == strongest_peaks_index:
                # mark the strongest peaks in a thicker manner
                plt.plot(
                    peaks[i],
                    y_coords[peaks[i]],
                    marker=next(marker),
                    linestyle="None",
                    markersize=8,
                    color="black",
                )
            else:
                plt.plot(
                    peaks[i],
                    y_coords[peaks[i]],
                    marker=next(marker),
                    linestyle="None",
                    markersize=5,
                )
        plt.legend()
        plt.show()


def bpm_from_beats(beats):
    if beats.size == 0:
        return 0
    m_res = scipy.stats.linregress(numpy.arange(len(beats)), beats)

    first_beat = m_res.intercept
    beat_step = m_res.slope

    return 60 / beat_step


def bops_realistic_smoothing(bops, min_spacing):
    return bops[numpy.where(numpy.diff(bops) > min_spacing)[0]]

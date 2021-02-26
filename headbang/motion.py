import numpy
import sys
import os
import scipy
from scipy.signal import find_peaks, peak_prominences
import matplotlib.pyplot as plt
from defaultlist import defaultlist
from headbang.params import DEFAULTS

openpose_install_path = "/home/sevagh/thirdparty-repos/openpose"

openpose_dir = openpose_install_path
sys.path.append(openpose_dir + "/build/python/openpose")

import pyopenpose as op


class OpenposeDetector:
    undef_coord_default = numpy.nan

    def __init__(
        self,
        n_frames,
        keypoints=DEFAULTS["pose_keypoints"],
        obj_limit=DEFAULTS["detected_object_limit"],
        adaptive_prominence_ratio=DEFAULTS["adaptive_prominence_ratio"],
        openpose_confidence_threshold=DEFAULTS["openpose_confidence_thresh"],
        peak_width=DEFAULTS["peak_width"],
    ):
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

        self.keypoints = [int(i) for i in keypoints.split(",")]

        self.n_frames = int(n_frames)
        self.all_y_coords = [[OpenposeDetector.undef_coord_default] * self.n_frames]
        self.frame_idx = 0
        self.obj_limit = obj_limit

        self.confidence_threshold = openpose_confidence_threshold
        self.adaptive_prominence_ratio = adaptive_prominence_ratio
        self.peak_width = peak_width

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
                    if i in self.keypoints and d[2] > self.confidence_threshold
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
                            self.all_y_coords.append(
                                [OpenposeDetector.undef_coord_default] * self.n_frames
                            )
                            self.all_y_coords[i][self.frame_idx] = y_norm

        self.frame_idx += 1
        return outframe

    def find_peaks(self):
        peaks = [None] * len(self.all_y_coords)
        prominences = [None] * len(self.all_y_coords)
        adjusted_y_coords = [None] * len(self.all_y_coords)

        for i, y_coords in enumerate(self.all_y_coords):
            min_coord = numpy.nanmin(y_coords)
            max_coord = numpy.nanmax(y_coords)

            # adaptive peak prominence - X% of max displacement
            adaptive_prominence = self.adaptive_prominence_ratio * (
                max_coord - min_coord
            )

            adjusted_y_coords[i] = numpy.nan_to_num(y_coords, nan=min_coord)

            peaks[i], _ = find_peaks(
                adjusted_y_coords[i],
                prominence=adaptive_prominence,
                wlen=self.peak_width,
            )

            prominences[i], _, _ = peak_prominences(adjusted_y_coords[i], peaks[i])

        top_ycoords_and_peaks = [
            (ycrds, pks)
            for _, pks, ycrds in sorted(
                zip(prominences, peaks, adjusted_y_coords),
                key=lambda triplet: sum(triplet[0]),
                reverse=True,
            )
        ]

        # only track up to obj_limit objects
        return top_ycoords_and_peaks[: self.obj_limit]

    def plot_ycoords(self):
        plt.figure(1)
        plt.title("normalized y coordinate motion")

        plt.xlabel("frame")
        plt.ylabel("y coord")

        frames = numpy.arange(self.n_frames)
        best_coords_and_peaks = self.find_peaks()

        for i, coordspeaks in enumerate(best_coords_and_peaks):
            y_coords, peaks = coordspeaks
            y_coords = numpy.asarray(y_coords)
            plt.plot(
                frames,
                y_coords,
                "-D",
                label="obj {0}".format(i),
                markevery=peaks,
                mec="black",
                mfc="black",
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

from cProfile import label
from models.networks import PointFlow
import os
import torch
import numpy as np
import torch.nn as nn
from args import get_args
import open3d
import glob
import transforms3d
import json


def load_pc_points(pcd_path):
    # return points in numpy array
    pcd = open3d.io.read_point_cloud(pcd_path)
    np_arr_pts = np.asarray(pcd.points).reshape(1, -1, 3)
    return np_arr_pts


def get_file_stem(path):
    return os.path.basename(path).split(".")[0]


def crop_points(full_pc_points, bboxpvrcnn_labels):
    # todo: crop points within labels and return a list of object points
    pass


def preprocessing(obj_points_list, bboxpvrcnn_labels, resize_flag, rotate_flag):
    # todo: normalize (translate to origin and rotate points) object points
    # todo: resize point cloud to 1/10 if resize_flag set
    # todo: rotate point cloud if rotate_flag set
    pass


def post_processing(obj_points_list, bboxpvrcnn_labels, resize_flag, rotate_flag):
    # todo: rotate to original orientation if rotate flag set
    # todo: resize back to original size if resize_flag set
    # todo: place object points back to original location
    pass


def main(args):
    assert len(glob.glob(args.label_dir + "/*.json")) == len(
        glob.glob(args.pcd_dir + "/*.pcd")
    ), "Amount of labels does not match the amount of point clouds"

    if args.output_dir != None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    for label_path in glob.glob(args.label_dir + "/*.json"):
        file_stem = get_file_stem(label_path)
        pcd_path = args.pcd_dir + "/{}.pcd".format(file_stem)
        if not os.path.exists(pcd_path):
            print("pcd_path {} does not exist.".format(pcd_path))
            exit(1)
        full_pc_points = load_pc_points(pcd_path)
        with open(label_path, "r") as fp:
            bboxpvrcnn_labels = json.load(fp)
        crop_points(full_pc_points, bboxpvrcnn_labels)


if __name__ == "__main__":
    args = get_args()
    main(args)

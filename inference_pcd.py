from cProfile import label
from models.networks import PointFlow
import os
import torch
import numpy as np
import torch.nn as nn
from args import get_args
import open3d
import glob
import json
from tqdm import tqdm


def load_pc_points(pcd_path):
    # return points in numpy array
    pcd = open3d.io.read_point_cloud(pcd_path)
    np_arr_pts = np.asarray(pcd.points)
    return np_arr_pts


def get_file_stem(path):
    return os.path.basename(path).split(".")[0]


def bboxpvrcnn_to_oriented_bboxes(bboxpvrcnn_labels):
    # Select class `Car` and transform to a list of oriented_bboxes
    oriented_bboxes = list()
    for bboxpvrcnn in bboxpvrcnn_labels:
        if bboxpvrcnn["class_name"] == "Car":
            # print("Car")
            euler_rot = np.array([0.0, 0.0, float(bboxpvrcnn["heading"])])
            rotation_mat = open3d.geometry.get_rotation_matrix_from_xyz(euler_rot)
            center_pos = np.array(
                [float(bboxpvrcnn["x"]), float(bboxpvrcnn["y"]), float(bboxpvrcnn["z"])]
            )
            extent = np.array(
                [
                    float(bboxpvrcnn["dx"]),
                    float(bboxpvrcnn["dy"]),
                    float(bboxpvrcnn["dz"]),
                ]
            )
            oriented_box = open3d.geometry.OrientedBoundingBox(
                center_pos, rotation_mat, extent
            )
            oriented_bboxes.append(oriented_box)
    return oriented_bboxes


def crop_points(full_pc_points, oriented_bboxes):
    # crop points within labels and return a list of object points
    obj_point_list = list()
    # print(full_pc_points.shape)
    vec3d_full_pc_points = open3d.utility.Vector3dVector(full_pc_points)
    for oriented_bbox in oriented_bboxes:
        # print(oriented_bbox.get_center())
        # print(oriented_bbox.extent)
        # print(oriented_bbox.get_max_bound())
        # print(oriented_bbox.get_min_bound())
        point_indices_in_box = oriented_bbox.get_point_indices_within_bounding_box(
            vec3d_full_pc_points
        )
        # print(point_indices_in_box)
        obj_point_list.append(full_pc_points[point_indices_in_box])
    return obj_point_list


def preprocessing(obj_points_list, oriented_bboxes, resize_flag, rotate_flag):
    # todo: normalize (translate to origin and rotate points) object points
    # todo: resize point cloud to 1/10 if resize_flag set
    # todo: rotate point cloud if rotate_flag set
    pass


def post_processing(obj_points_list, oriented_bboxes, resize_flag, rotate_flag):
    # todo: rotate to original orientation if rotate flag set
    # todo: resize back to original size if resize_flag set
    # todo: place object points back to original location
    pass


def save_pcd(np_arr_points, output_path):
    # save np_arr_points into pcd file
    vec3d_points = open3d.utility.Vector3dVector(np_arr_points)
    pcd = open3d.geometry.PointCloud(vec3d_points)
    open3d.io.write_point_cloud(output_path, pcd)


def main(args):
    assert len(glob.glob(args.labels_dir + "/*.json")) == len(
        glob.glob(args.pcd_dir + "/*.pcd")
    ), "Amount of labels does not match the amount of point clouds"

    if args.output_dir != None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    for label_path in tqdm(sorted(glob.glob(args.labels_dir + "/*.json"))):
        file_stem = get_file_stem(label_path)
        pcd_path = args.pcd_dir + "/{}.pcd".format(file_stem)
        if not os.path.exists(pcd_path):
            print("pcd_path {} does not exist.".format(pcd_path))
            exit(1)
        full_pc_points = load_pc_points(pcd_path)
        with open(label_path, "r") as fp:
            bboxpvrcnn_labels = json.load(fp)
        oriented_bboxes = bboxpvrcnn_to_oriented_bboxes(bboxpvrcnn_labels)
        # print(len(oriented_bboxes))
        # print(len(bboxpvrcnn_labels))
        obj_points_list = crop_points(full_pc_points, oriented_bboxes)
        # todo: Verify crop points by saving object points to pcd
        for i, obj_points in enumerate(obj_points_list):
            if i == 0:
                obj_points_accm = obj_points
            else:
                obj_points_accm = np.concatenate((obj_points_accm, obj_points))

        if args.output_dir != None:
            output_path = "{}/{}.pcd".format(args.output_dir, file_stem)
            save_pcd(obj_points_accm, output_path)


if __name__ == "__main__":
    args = get_args()
    main(args)

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


def single_rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """

    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros_like(angle)
    ones = np.ones_like(angle)
    rot_matrix = np.stack(
        (cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), axis=0
    ).reshape(3, 3)
    points_rot = np.matmul(points[:, 0:3], rot_matrix)
    return points_rot


def rotate_points_by_rot_mat(points, rot_matrix):
    points_rot = np.matmul(points[:, 0:3], rot_matrix)
    return points_rot


def get_normalized_cloud(pnts, gt_box):
    """
    Return points with their `(x, y, z)` rotated with -heading, i.e., all lined up to `0 degree`, also with their bottom filtered.
    """
    pnts[:, :3] -= gt_box[:3]
    pnts = np.concatenate(
        [single_rotate_points_along_z(pnts[:, :3], -gt_box[6]), pnts[:, 3:]], axis=1
    )
    return pnts


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
    gt_bboxes_list = list()
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
            gt_box = [
                float(bboxpvrcnn["x"]),
                float(bboxpvrcnn["y"]),
                float(bboxpvrcnn["z"]),
                float(bboxpvrcnn["dx"]),
                float(bboxpvrcnn["dy"]),
                float(bboxpvrcnn["dz"]),
                float(bboxpvrcnn["heading"]),
            ]
            gt_bboxes_list.append(gt_box)
            oriented_bboxes.append(oriented_box)
    return oriented_bboxes, gt_bboxes_list


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


def preprocessing(
    obj_points_list, oriented_bboxes, gt_bboxes_list, resize_flag, rotate_flag
):
    # normalize (translate to origin and rotate points) object points

    norm_obj_points_list = list()
    for i, obj_points in enumerate(obj_points_list):
        norm_obj_points = get_normalized_cloud(obj_points, gt_bboxes_list)
        norm_obj_points_list.append(norm_obj_points)
    # resize point cloud to 1/10 if resize_flag set
    if resize_flag:
        resized_list = list()
        for obj_points in norm_obj_points_list:
            resize_obj_points = np.divide(obj_points, 10.0)
            resized_list.append(resize_obj_points)
        norm_obj_points_list = resized_list

    # rotate point cloud if rotate_flag set
    if rotate_flag:
        euler_rot = np.array([-np.pi / 2, 0.0, np.pi / 2])
        rotation_mat = open3d.geometry.get_rotation_matrix_from_xyz(euler_rot)
        rotated_list = list()
        for i, obj_points in enumerate(norm_obj_points_list):
            rotated_obj_points = rotate_points_by_rot_mat(obj_points, rotation_mat)
            rotated_list.append(rotated_obj_points)
        norm_obj_points_list = rotated_list

    return norm_obj_points


def post_processing(
    obj_points_list, oriented_bboxes, gt_bboxes_list, resize_flag, rotate_flag
):
    # todo: rotate to original orientation if rotate flag set
    # todo: resize back to original size if resize_flag set
    # todo: place object points back to original location
    pass


def create_batched_data(processed_point_list):
    # todo: turn processed_point_list of length N into (N, M, 3) torch tensor for model loading
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
        oriented_bboxes, gt_bboxes_list = bboxpvrcnn_to_oriented_bboxes(
            bboxpvrcnn_labels
        )
        # print(len(oriented_bboxes))
        # print(len(bboxpvrcnn_labels))
        obj_points_list = crop_points(full_pc_points, oriented_bboxes)
        assert len(obj_points_list) == len(oriented_bboxes)
        # Verify crop points by saving object points to pcd
        # for i, obj_points in enumerate(obj_points_list):
        #     if i == 0:
        #         obj_points_accm = obj_points
        #     else:
        #         obj_points_accm = np.concatenate((obj_points_accm, obj_points))
        # if args.output_dir != None:
        #     output_path = "{}/{}.pcd".format(args.output_dir, file_stem)
        #     save_pcd(obj_points_accm, output_path)
        processed_point_list = preprocessing(
            obj_points_list,
            oriented_bboxes,
            gt_bboxes_list,
            args.resize_flag,
            args.rotate_flag,
        )


if __name__ == "__main__":
    args = get_args()
    main(args)

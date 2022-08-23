import open3d as o3d
import numpy as np
import os
import glob
import argparse

car_cat_id = "02958343"


def npy_to_pcd(np_arr_points):
    vec3d_points = o3d.utility.Vector3dVector(np_arr_points)
    pcd = o3d.geometry.PointCloud(vec3d_points)
    return pcd


def check_dataset(dataset_path):
    car_set_dir_path = dataset_path + "/" + car_cat_id
    if not os.path.exists(car_set_dir_path):
        print("{} does not exists.".format(car_set_dir_path))
        exit(1)
    if not os.path.isdir(car_set_dir_path):
        print("{} is not a directory.".format(car_set_dir_path))
        exit(1)
    return car_set_dir_path


def main(args):
    car_set_dir_path = check_dataset(args.input_dir)
    for dir in glob.glob("{}/*/".format(car_set_dir_path)):
        sub_dir = os.path.basename(os.path.normpath(dir))
        output_sub_dir = "{}/{}".format(args.output_dir, sub_dir)
        if not os.path.exists(output_sub_dir):
            print("Creating {}".format(output_sub_dir))
            os.makedirs(output_sub_dir)
        # print(dir)
        for npy_path in glob.glob("{}/*.npy".format(dir)):
            # print(npy_path)
            np_arr_points = np.load(npy_path)
            pcd = npy_to_pcd(np_arr_points)
            npy_file_stem = os.path.basename(npy_path).split(".")[0]
            output_pcd_path = "{}/{}.pcd".format(output_sub_dir, npy_file_stem)
            o3d.io.write_point_cloud(output_pcd_path, pcd)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Transforms ShapeNetCore dataset stored in npy format to pcd format",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        required=True,
        help="input directory to the ShapeNet dataset",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="output directory to store the pcd files",
    )

    args = parser.parse_args()
    if not os.path.exists(args.input_dir):
        print("input_dir doesn't exists!")
        exit(1)

    if not os.path.isdir(args.input_dir):
        print("input_dir is not a directory!")
        exit(1)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)


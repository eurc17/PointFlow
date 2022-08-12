from models.networks import PointFlow
import os
import torch
import numpy as np
import torch.nn as nn
from args import get_args
import open3d


def main(args):
    model = PointFlow(args)

    def _transform_(m):
        return nn.DataParallel(m)

    model = model.cuda()
    model.multi_gpu_wrapper(_transform_)

    pcd = open3d.io.read_point_cloud("./tools/pcds/10000_5.pcd")
    np_arr_pts = np.asarray(pcd.points).reshape(1, -1, 3)
    print(np_arr_pts.shape)
    torch_pts = torch.from_numpy(np_arr_pts).float()

    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint = torch.load(args.resume_checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()
    N = 2048

    with torch.no_grad():
        out_pc = model.reconstruct(torch_pts, num_points=N)
    out_pc_npy = out_pc.detach().cpu().numpy().reshape(-1, 3)
    vec_3d_pts = open3d.utility.Vector3dVector(out_pc_npy)
    out_pcd = open3d.geometry.PointCloud(vec_3d_pts)
    print(out_pc_npy.shape)

    open3d.io.write_point_cloud("./tools/gen_pcds/10000_5.pcd", out_pcd)


if __name__ == "__main__":
    args = get_args()
    main(args)

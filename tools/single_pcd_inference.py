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
    # model.multi_gpu_wrapper(_transform_)

    pcd = open3d.io.read_point_cloud("./pcds/10000_1.pcd")
    np_arr_pts = np.asarray(pcd.points)

    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint = torch.load(args.resume_checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()
    N = 2048

    with torch.no_grad():
        out_pc = model.reconstruct(np_arr_pts, num_points=N)
    open3d.io.write_point_cloud("./gen_pcds/10000_1.pcd", out_pc)


if __name__ == "__main__":
    args = get_args()
    main(args)

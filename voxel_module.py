# This file is modified from https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/ops/voxel/voxelize.py

import torch
import torch.nn as nn
from voxel_op import hard_voxelize
import numpy as np
import time
class _Voxelization(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                points,
                voxel_sizes, 
                coors_range,
                max_pointses, 
                max_voxels=20000, 
                deterministic=True): 
        """convert kitti points(N, >=3) to voxels.
        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
                size
            coors_range: [6] list/tuple or array, float. indicate voxel
                range. format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.
            deterministic: bool. MUST BE OPENED. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        Returns:
            voxels: [scale_num, M, max_points, ndim] float tensor. only contain points
                    and returned when max_points != -1.
            coordinates: [scale_num, M, 3] int32 tensor, always returned.
            num_points_per_voxel: [scale_num, M] int32 tensor. Only returned when
                max_points != -1.
        """
        assert len(voxel_sizes) == len(max_pointses), 'Scale num must match the points num limit. Please assign points num limit for every scale'
        assert deterministic, 'deterministic must be opened. And the non deterministic version will be coming soon.'
        max_points_for_all_scale = max(max_pointses)
        scale_num = len(voxel_sizes)
        voxels = points.new_zeros(
            size=(scale_num, max_voxels, max_points_for_all_scale, points.size(1)))
        coors = points.new_zeros(size=(scale_num, max_voxels, 3), dtype=torch.int)
        num_points_per_voxel = points.new_zeros(
            size=(scale_num, max_voxels), dtype=torch.int)
        voxel_num = hard_voxelize(points, voxels, coors,
                                    num_points_per_voxel, voxel_sizes,
                                    coors_range, max_pointses, max_voxels, 3,
                                    deterministic)
        voxels_out, coors_out, num_points_per_voxel_out = [], [], []
        for i in range(scale_num):
            voxels_out.append(voxels[i, :voxel_num[i], :max_pointses[i]])
            coors_out.append(coors[i, :voxel_num[i], :max_pointses[i]].flip(-1))
            num_points_per_voxel_out.append(num_points_per_voxel[i, :voxel_num[i]])
        return voxels_out, coors_out, num_points_per_voxel_out

def load_point_from_bin(pcd_bin_path, device):
    points = np.fromfile(pcd_bin_path, dtype=np.float32).reshape(-1, 4)
    points_tensor = torch.tensor(points, dtype=torch.float32)
    points_tensor = points_tensor.to(device)
    return points_tensor

if __name__ == '__main__':

    ################# QUICK TEST #################
    voxel_sizes1 = [[0.16, 0.16, 4] for i in range(100)]
    voxel_sizes2 = [[0.32, 0.32, 4] for i in range(100)]
    voxel_sizes3 = [[0.64, 0.64, 4] for i in range(100)]
    point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
    max_pointses1 = [32 for i in range(100)]
    max_pointses2 = [64 for i in range(100)]
    max_pointses3 = [128 for i in range(100)]
    max_voxels = 20000
    pcd_bin_path = '/root/PointPillars/ops/1618798396099.bin'
    assert pcd_bin_path == '/root/PointPillars/ops/1618798396099.bin', 'please load your own point clouds'
    device = 'cuda:2'
    time0 = time.time()
    input_pcd_tensor = load_point_from_bin(pcd_bin_path, device)
    
    for i in range(100):
        voxels_out, coors_out, num_points_per_voxel_out = _Voxelization.apply(input_pcd_tensor, [voxel_sizes1[i]], point_cloud_range,
                                   [max_pointses1[i]], max_voxels)
    for i in range(100):
        voxels_out, coors_out, num_points_per_voxel_out = _Voxelization.apply(input_pcd_tensor, [voxel_sizes2[i]], point_cloud_range,
                                [max_pointses2[i]], max_voxels)
    for i in range(100):
        voxels_out, coors_out, num_points_per_voxel_out = _Voxelization.apply(input_pcd_tensor, [voxel_sizes3[i]], point_cloud_range,
                                   [max_pointses3[i]], max_voxels)
    time1 = time.time()
    for i in range(100):
        voxels_out, coors_out, num_points_per_voxel_out = _Voxelization.apply(input_pcd_tensor, [voxel_sizes1[i], voxel_sizes2[i],voxel_sizes3[i]], point_cloud_range,
                            [max_pointses1[i], max_pointses2[i], max_pointses3[i]], max_voxels)
    time2 = time.time()

    print(f'time1:{time1 - time0}; time2:{time2 - time1}')
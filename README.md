# Multiscale_Voxelization_CUDA
Voxelization of LiDAR point cloud is a classical method for feature extraction preprocess, and implementation on GPU will significantly improve the speed🚀. The ordinary voxelization CUDA operator only provides voxel input of a single scale, which lacks convenience in its invocation. 

This is a **multiscale voxelization operator for CUDA implementation**🥪. 

Building upon the work of normal voxelization method, it introduces the functionality to specify multiple voxel scales. The python API has been simplified, and upon testing with an NVIDIA RTX 3090, it shows a 10% speed improvement.
## Get Started
You can get started with the following steps:
```shell
git clone https://github.com/RoachNier/Multiscale_Voxelization_CUDA.git
cd Multiscale_Voxelization_CUDA
python setup.py develop
```
Finally, feel free to run the test python script or integrate it into your own network.

## How to Use
After compile the operator to a shared object(.so on Linux), you can use it in your python script. You can refer it in ```voxel_module.py```.

For example, we simply explain the test script in ```voxel_module.py```:
```python
# import the function from the compiled .so file
from voxel_op import hard_voxelize
# define voxel sizes which MUST BE a nested List
voxel_sizes = [[0.16, 0.16, 4], [0.32, 0.32, 4], [0.64, 0.64, 4]]
# define points number limit for every voxel size scale
max_points_for_all_scales = [32, 64, 128]
# define point cloud range by meter unit
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
# define voxel number limit for all scales
max_voxels = 20000
# load your own point cloud data as torch.tensor and put it to CUDA
pcd_bin_path = 'your/point cloud bin/path'
device = 'cuda:2'
input_pcd_tensor = load_point_from_bin(pcd_bin_path, device)
# get your multiscale voxelization feature output
voxels_out, coors_out, num_points_per_voxel_out = _Voxelization.apply(input_pcd_tensor, voxel_sizes, point_cloud_range,
                            max_points_for_all_scales, max_voxels)
```

## TODO List
- [ ] Non deterministic optimization
- [ ] Batch implementation

## Acknowledgement
Thanks for the open souce code [mmcv](https://github.com/open-mmlab/mmcv), [mmdet](https://github.com/open-mmlab/mmdetection) and [mmdet3d](https://github.com/open-mmlab/mmdetection3d). And Thanks for our contributor [RunkaiZhao]().

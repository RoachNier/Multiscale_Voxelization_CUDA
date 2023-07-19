# Multiscale_Voxelization_CUDA
This is a multiscale voxelization operator for CUDA implementation. 

Building upon the work of normal voxelization method, it introduces the functionality to specify multiple voxel scales. The python API has been simplified, and upon testing with an Nvidia RTX 3090, it shows a 10% speed improvement.
## Get Started
You can get started with the following steps:
```shell
git clone https://github.com/RoachNier/Multiscale_Voxelization_CUDA.git
cd Multiscale_Voxelization_CUDA
python setup.py develop
```
Finally, feel free to run the test python script or integrate it into your own network.

## Acknowledgement
Thanks for the open souce code [mmcv](https://github.com/open-mmlab/mmcv), [mmdet](https://github.com/open-mmlab/mmdetection) and [mmdet3d](https://github.com/open-mmlab/mmdetection3d).

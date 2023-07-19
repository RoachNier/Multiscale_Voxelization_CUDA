from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointpillars',
    ext_modules=[
        CUDAExtension(
            name='voxel_op', 
            sources=['voxelization/voxelization.cpp',
                     'voxelization/voxelization_cuda.cu',
                    ],
            define_macros=[('WITH_CUDA', None)]    
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
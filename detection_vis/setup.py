"""Setup script for DetZero vision detection module."""

import os
import subprocess
from setuptools import find_packages, setup


def get_git_commit_number():
    if not os.path.exists('../.git'):
        return '0000000'
    
    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == '__main__':
    version = '0.1.0+%s' % get_git_commit_number()
    write_version_to_file(version, 'detzero_det_vis/version.py')
    
    setup(
        name='detzero_det_vis',
        version=version,
        description='Vision-based 3D object detection module for DetZero framework using BEVFormer',
        install_requires=[
            'numpy',
            'torch>=1.8.0',
            'torchvision',
            'numba',
            'tensorboardX',
            'easydict',
            'pyyaml',
            'scikit-image',
            'tqdm',
            'opencv-python',
            'Pillow',
        ],
        author='PJLab-ADLab',
        license='Apache License 2.0',
        packages=find_packages(),
        cmdclass={},
    )


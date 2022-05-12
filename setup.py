# this code from MobileViTV2
# https://github.com/apple/ml-cvnets/blob/d38a116fe134a8cd5db18670764fdaafd39a5d4f/setup.py

from setuptools import find_packages, setup
import sys
import os

VERSION = 0.1

def do_setup(package_data):
    setup(
        name="TransCycle model",
        version=VERSION,
        description="TransCycleNet: A library for training computer vision networks",
        url="https://github.com/Qqww4599/Prososal-Attention-Model",
        setup_requires=[
            'numpy<1.20.0; python_version<"3.7"',
            'numpy; python_version>="3.7"',
            "setuptools>=18.0",
        ],
        install_requires=[
            'numpy<1.20.0; python_version<"3.7"',
            'numpy; python_version>="3.7"',
            "torch",
            "tqdm",
        ],
        packages=find_packages(
            exclude=[
                "train_config",
                "train_config.*"
            ]
        ),
        package_data=package_data,
        test_suite="tests",
        entry_points={
            "console_scripts": [
                "TransCycleNet-train = main_train:main_worker",
                "TransCycleNet-eval = main_eval:main_worker",
                "TransCycleNet-eval-seg = main_eval:main_worker_segmentation",
                "TransCycleNet-eval-det = main_eval:main_worker_detection",
                "TransCycleNet-convert = main_conversion:main_worker_conversion"
            ],
        },
        zip_safe=False,
    )

def get_files(path, relative_to="."):
    all_files = []
    for root, _dirs, files in os.walk(path, followlinks=True):
        root = os.path.relpath(root, relative_to)
        for file in files:
            if file.endswith(".pyc"):
                continue
            all_files.append(os.path.join(root, file))
    return all_files


if __name__ == "__main__":
    package_data = {
        "cvnets": (
            get_files(os.path.join("MainResearch", "config"))
        )
    }
    do_setup(package_data)
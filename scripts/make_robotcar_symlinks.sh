#!/bin/bash
#
#Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
#
 
set -x
# change the following line
ROBOTCAR_SDK_ROOT=/data/robotcar-dataset-sdk

ln -s ${ROBOTCAR_SDK_ROOT}/models/ ../data/robotcar_camera_models
ln -s ${ROBOTCAR_SDK_ROOT}/python/ ../dataset_loaders/robotcar_sdk
set +x

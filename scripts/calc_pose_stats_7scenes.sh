#!/usr/bin/env bash
#
#Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
#
 
scenes=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")

set -x
for scene in "${scenes[@]}"
do
  python calc_pose_stats.py --dataset 7Scenes --scene ${scene}
done
set +x

#!/usr/bin/env bash
#
#Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
#
 
scenes=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")
n_seqs=(6 4 2 10 8 14 6)
source ~/miniconda2/bin/activate pytorch
PYTHON=python

for ((i=0;i<${#scenes[@]};++i));
do
    scene=${scenes[i]}

    for ((seq=1;seq<=${n_seqs[i]};++seq));
    do
        set -x
        $PYTHON align_vo_poses.py --dataset 7Scenes --scene $scene --subsample 10 --seq $seq --vo_lib dso
        set +x
    done
done

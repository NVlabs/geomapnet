#!/usr/bin/env bash
#
#Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
#
 
DATA_ROOT=/media/TB/7scenes
scenes=( "chess" "fire" "office" "redkitchen" "heads" "pumpkin" "stairs" )

for scene in "${scenes[@]}"
do
    unzip ${DATA_ROOT}/${scene}.zip -d ${DATA_ROOT}
    rm ${DATA_ROOT}/${scene}.zip

    for seq in ${DATA_ROOT}/${scene}/*.zip
    do
        unzip ${seq} -d ${DATA_ROOT}/${scene}/
        rm ${seq}

    done
done

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
"""
Generates a GPS VO file for the RobotCar dataset
"""

import argparse
import os.path as osp
import csv

parser = argparse.ArgumentParser('GPS INS generation script')
parser.add_argument('--scene', required=True, type=str,
                    help='Scene identifier')
parser.add_argument('--seq', required=True, type=str,
                    help='Sequence identifier')
args = parser.parse_args()

data_dir = osp.join('..', 'data', 'deepslam_data', 'RobotCar', args.scene,
                    args.seq)
gps_file = osp.join(data_dir, 'gps', 'gps.csv')
out_file = osp.join(data_dir, 'gps', 'gps_ins.csv')

with open(gps_file, 'r') as fin, open(out_file, 'w') as fout:
  reader = csv.reader(fin)
  writer = csv.writer(fout)
  next(fin)
  header = 'timestamp,ins_status,latitude,longitude,altitude,northing,easting,' \
           'down,utm_zone,velocity_north,velocity_east,velocity_down,roll,' \
           'pitch,yaw\n'
  fout.write(header)
  for inrow in reader:
    outrow = []
    outrow.append(inrow[0])  # timestamp
    outrow.append('INS_SOLUTION_GOOD')  # ins_status
    outrow.append(inrow[2])  # latitude
    outrow.append(inrow[3])  # longitude
    outrow.append(inrow[4])  # altitude
    outrow.append(inrow[8])  # northing
    outrow.append(inrow[9])  # easting
    outrow.append(inrow[10]) # down
    outrow.append(inrow[11]) # utm_zone
    outrow.extend([0,0,0,0,0,0])
    writer.writerow(outrow)
print '{:s} written'.format(out_file)

"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
 
import timeit
import numpy as np

repeat = 5
number = 50

setup = ''
setup += 'import utils;'
setup += 'und_im_filename = "/media/TB/test.png";'
t = timeit.Timer(stmt='utils.load_image(und_im_filename)', setup=setup)
times = t.repeat(repeat=repeat, number=number)
times = np.asarray(times)
print 'Normal image load'
print 'Times = ', times
print 'min = {:8.7f} seconds'.format(np.min(times))

setup = ''
setup += 'from robotcar_sdk.image import load_image;'
setup += 'from robotcar_sdk.camera_model import CameraModel;'
setup += 'c = CameraModel("../data/robotcar_camera_models/", "stereo/centre");'
setup += 'raw_im_filename = "/media/TB/deepslam_data/RobotCar/overcast/2015-08-14-14-54-57/stereo/centre/1439560497946501.png";'
t = timeit.Timer(stmt='load_image(raw_im_filename)', setup=setup)
times = t.repeat(repeat=repeat, number=number)
times = np.asarray(times)
print 'Demosaic image load'
print 'Times = ', times
print 'min = {:8.7f} seconds'.format(np.min(times))

t = timeit.Timer(stmt='load_image(raw_im_filename, model=c)', setup=setup)
times = t.repeat(repeat=repeat, number=number)
times = np.asarray(times)
print 'Demosaic + undistort image load'
print 'Times = ', times
print 'min = {:8.7f} seconds'.format(np.min(times))

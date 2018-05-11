## License 

Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

## Notes for Writing Data Loader

Follow the style used in `dataset_loaders/robotcar.py` and
`dataset_loaders/seven_scenes.py`. Importantly, the dataloader will need to
define the `self.gt_idx` attribute for use with the higher-level datasets in
`dataset_loaders/composite.py`.


The `gt_idx` attribute is used when `real == True`
(i.e. some SLAM/integration of VO is used to get poses for the images) and the
VO library skips some images. In this case, length of the dataset will be
smaller than if `real == False` (i.e. using the dataset-provided ground truth
poses). `gt_idx` is a list whose i-th element is the index of the i-th image
if it were part of an instance of this class with `real == False`.

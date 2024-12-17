# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MSDDataset(BaseSegDataset):
    """PhoneScreenDataset dataset.
    
     In segmentation map annotation for PhoneScreenDataset, 0 stands for background,
    which is included in 4 categories. ``reduce_zero_label`` is fixed to False.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes = (
        'background', 'oil', 'scratch',  'stain'),
        palette = [[80, 80, 80], [180, 120, 120], [6, 230, 230], [154, 255, 154]] 
        )
    

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

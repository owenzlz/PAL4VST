from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class ArtifactsDataset(BaseSegDataset):

    METAINFO = dict(
        classes=('background', 'bad_region'),
        palette=[[0, 0, 0], [255, 0, 255]])

    def __init__(self, **kwargs):
        super(ArtifactsDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', **kwargs)






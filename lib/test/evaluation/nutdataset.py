import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class NUTDataset(BaseDataset):
    """ NAT2021L dataset.
    """

    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.nut_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                           sequence_path=sequence_path, frame=frame_num,
                                                                           nz=nz, ext=ext) for frame_num in
                  range(start_frame + init_omit, end_frame + 1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'nut', ground_truth_rect[init_omit:, :],
                        object_class=sequence_info['object_class'])

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {'name': 'car_l4', 'path': 'data_seq/car_l4', 'startFrame': 1, 'endFrame': 2558, 'nz': 5,
              'ext': 'jpg', 'anno_path': 'anno/car_l4.txt', 'object_class': 'other'},
             {'name': 'car_l1', 'path': 'data_seq/car_l1', 'startFrame': 1, 'endFrame': 3234, 'nz': 5,
              'ext': 'jpg', 'anno_path': 'anno/car_l1.txt', 'object_class': 'other'},
             {'name': 'signpost2', 'path': 'data_seq/signpost2', 'startFrame': 1, 'endFrame': 2013, 'nz': 5,
              'ext': 'jpg', 'anno_path': 'anno/signpost2.txt', 'object_class': 'other'},
             {'name': 'girl3', 'path': 'data_seq/girl3', 'startFrame': 1, 'endFrame': 1731, 'nz': 5,
              'ext': 'jpg', 'anno_path': 'anno/girl3.txt', 'object_class': 'other'},
             {'name': 'N04003', 'path': 'data_seq/N04003', 'startFrame': 1, 'endFrame': 2254, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N04003.txt', 'object_class': 'other'},
             {'name': 'N04007', 'path': 'data_seq/N04007', 'startFrame': 1, 'endFrame': 2332, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N04007.txt', 'object_class': 'other'},
             {'name': 'N08004', 'path': 'data_seq/N08004', 'startFrame': 1, 'endFrame': 2648, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N08004.txt', 'object_class': 'other'},
             {'name': 'signpost4', 'path': 'data_seq/signpost4', 'startFrame': 1, 'endFrame': 1466, 'nz': 5,
              'ext': 'jpg', 'anno_path': 'anno/signpost4.txt', 'object_class': 'other'},
             {'name': 'N09001', 'path': 'data_seq/N09001', 'startFrame': 1, 'endFrame': 3866, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N09001.txt', 'object_class': 'other'},
             {'name': 'bike11', 'path': 'data_seq/bike11', 'startFrame': 1, 'endFrame': 1981, 'nz': 5,
              'ext': 'jpg', 'anno_path': 'anno/bike11.txt', 'object_class': 'other'},
             {'name': 'N08006', 'path': 'data_seq/N08006', 'startFrame': 1, 'endFrame': 2121, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N08006.txt', 'object_class': 'other'},
             {'name': 'N02005', 'path': 'data_seq/N02005', 'startFrame': 1, 'endFrame': 1511, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N02005.txt', 'object_class': 'other'},
             {'name': 'N08001', 'path': 'data_seq/N08001', 'startFrame': 1, 'endFrame': 2102, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N08001.txt', 'object_class': 'other'},
             {'name': 'N08005', 'path': 'data_seq/N08005', 'startFrame': 1, 'endFrame': 2671, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N08005.txt', 'object_class': 'other'},
             {'name': 'car_l3', 'path': 'data_seq/car_l3', 'startFrame': 1, 'endFrame': 3087, 'nz': 5,
              'ext': 'jpg', 'anno_path': 'anno/car_l3.txt', 'object_class': 'other'},
             {'name': 'N04006', 'path': 'data_seq/N04006', 'startFrame': 1, 'endFrame': 3193, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N04006.txt', 'object_class': 'other'},
             {'name': 'N08002', 'path': 'data_seq/N08002', 'startFrame': 1, 'endFrame': 1846, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N08002.txt', 'object_class': 'other'},
             {'name': 'N04008', 'path': 'data_seq/N04008', 'startFrame': 1, 'endFrame': 2275, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N04008.txt', 'object_class': 'other'},
             {'name': 'signpost5', 'path': 'data_seq/signpost5', 'startFrame': 1, 'endFrame': 1705, 'nz': 5,
              'ext': 'jpg', 'anno_path': 'anno/signpost5.txt', 'object_class': 'other'},
             {'name': 'car3', 'path': 'data_seq/car3', 'startFrame': 1, 'endFrame': 1419, 'nz': 5,
              'ext': 'jpg', 'anno_path': 'anno/car3.txt', 'object_class': 'other'},
             {'name': 'N04004', 'path': 'data_seq/N04004', 'startFrame': 1, 'endFrame': 1934, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N04004.txt', 'object_class': 'other'},
             {'name': 'car_l7', 'path': 'data_seq/car_l7', 'startFrame': 1, 'endFrame': 4571, 'nz': 5,
              'ext': 'jpg', 'anno_path': 'anno/car_l7.txt', 'object_class': 'other'},
             {'name': 'N04005', 'path': 'data_seq/N04005', 'startFrame': 1, 'endFrame': 1651, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N04005.txt', 'object_class': 'other'},
             {'name': 'girl6_2', 'path': 'data_seq/girl6_2', 'startFrame': 1, 'endFrame': 1600, 'nz': 5,
              'ext': 'jpg', 'anno_path': 'anno/girl6_2.txt', 'object_class': 'other'},
             {'name': 'person10_2', 'path': 'data_seq/person10_2', 'startFrame': 1, 'endFrame': 1621, 'nz': 5,
              'ext': 'jpg', 'anno_path': 'anno/person10_2.txt', 'object_class': 'other'},
             {'name': 'car_l6', 'path': 'data_seq/car_l6', 'startFrame': 1, 'endFrame': 3234, 'nz': 5,
              'ext': 'jpg', 'anno_path': 'anno/car_l6.txt', 'object_class': 'other'},
             {'name': 'bike6', 'path': 'data_seq/bike6', 'startFrame': 1, 'endFrame': 1623, 'nz': 5,
              'ext': 'jpg', 'anno_path': 'anno/bike6.txt', 'object_class': 'other'},
             {'name': 'car_l2', 'path': 'data_seq/car_l2', 'startFrame': 1, 'endFrame': 2248, 'nz': 5,
              'ext': 'jpg', 'anno_path': 'anno/car_l2.txt', 'object_class': 'other'},
             {'name': 'N03001', 'path': 'data_seq/N03001', 'startFrame': 1, 'endFrame': 1425, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N03001.txt', 'object_class': 'other'},
             {'name': 'N05001', 'path': 'data_seq/N05001', 'startFrame': 1, 'endFrame': 2429, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N05001.txt', 'object_class': 'other'},
             {'name': 'N02004', 'path': 'data_seq/N02004', 'startFrame': 1, 'endFrame': 3511, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N02004.txt', 'object_class': 'other'},
             {'name': 'N02003', 'path': 'data_seq/N02003', 'startFrame': 1, 'endFrame': 1446, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N02003.txt', 'object_class': 'other'},
             {'name': 'N02002', 'path': 'data_seq/N02002', 'startFrame': 1, 'endFrame': 3151, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N02002.txt', 'object_class': 'other'},
             {'name': 'N02001', 'path': 'data_seq/N02001', 'startFrame': 1, 'endFrame': 1679, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N02001.txt', 'object_class': 'other'},
             {'name': 'N04001', 'path': 'data_seq/N04001', 'startFrame': 1, 'endFrame': 2500, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N04001.txt', 'object_class': 'other'},
             {'name': 'N08003', 'path': 'data_seq/N08003', 'startFrame': 1, 'endFrame': 1847, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N08003.txt', 'object_class': 'other'},
             {'name': 'car_l5', 'path': 'data_seq/car_l5', 'startFrame': 1, 'endFrame': 2458, 'nz': 5,
              'ext': 'jpg', 'anno_path': 'anno/car_l5.txt', 'object_class': 'other'},
             {'name': 'N01001', 'path': 'data_seq/N01001', 'startFrame': 1, 'endFrame': 3391, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N01001.txt', 'object_class': 'other'},
             {'name': 'person3_1', 'path': 'data_seq/person3_1', 'startFrame': 1, 'endFrame': 1601, 'nz': 5,
              'ext': 'jpg', 'anno_path': 'anno/person3_1.txt', 'object_class': 'other'},
             {'name': 'signpost6', 'path': 'data_seq/signpost6', 'startFrame': 1, 'endFrame': 1515, 'nz': 5,
              'ext': 'jpg', 'anno_path': 'anno/signpost6.txt', 'object_class': 'other'},
             {'name': 'N04002', 'path': 'data_seq/N04002', 'startFrame': 1, 'endFrame': 1781, 'nz': 6,
              'ext': 'jpg', 'anno_path': 'anno/N04002.txt', 'object_class': 'other'},
             {'name': 'pedestrian_l', 'path': 'data_seq/pedestrian_l', 'startFrame': 1, 'endFrame': 2045,
              'nz': 5, 'ext': 'jpg', 'anno_path': 'anno/pedestrian_l.txt', 'object_class': 'other'}
        ]

        return sequence_info_list
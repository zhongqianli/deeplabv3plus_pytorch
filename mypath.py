class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return 'data/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return 'data/coco'
        elif dataset == 'fruit_seg':
            return "imgseg/dataset-494"
        elif dataset == 'iris_seg':
            return "iris_seg/CASIA-distance"
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

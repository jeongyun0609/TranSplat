from collections import defaultdict


class DatasetConfig:
    model_folder = 'models'
    train_folder = 'train'
    test_folder = 'test' 
    img_folder = 'rgb'
    depth_folder = 'depth'
    img_ext = 'png'
    depth_ext = 'png'


config = defaultdict(lambda *_: DatasetConfig())

config['tless'] = tless = DatasetConfig()
tless.model_folder = 'models_cad'
tless.test_folder = 'test_primesense'
tless.train_folder = 'train_primesense'

config['hb'] = hb = DatasetConfig()
hb.test_folder = 'test_primesense'

config['itodd'] = itodd = DatasetConfig()
itodd.depth_ext = 'png'
itodd.img_folder = 'rgb'
itodd.img_ext = 'png'
itodd.train_folder='train_pbr'
itodd.test_folder='train_pbr'


config['TRansPose'] = TRansPose = DatasetConfig()
TRansPose.train_folder = 'train'
TRansPose.test_folder = 'test'
TRansPose.depth_folder = 'depth/rendered'
TRansPose.img_folder = 'rgb'
TRansPose.mask_folder = 'rgb'

config = {
    "display_check": False,
    "train_check":False,
    "predict_check":True,
    "save_model":False,
    "IMG_ROWS":480,
    "IMG_COLS":320,
    "TEST_IMG_ROWS":1918,
    "TEST_IMG_COLS":1280,
    "model_path": "pretrained/carvana_model.h5",
    "test_img_path": "test_images/29bb3ece3180_11.jpg" 
}

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

opt = dotdict(config)
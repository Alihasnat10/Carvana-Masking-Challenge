from data_generation import extract_data, get_data_paths, get_train_test_data
from plot import display, plot_model_results
from model import train, load_model
from config import opt
import numpy as np
import imageio
import cv2
if __name__ == "__main__":
    if opt.train_check:
        _ = extract_data()
        train_img_paths, train_mask_paths = get_data_paths()
        NUM_EXAMPLES = len(train_img_paths)
        TRAIN_SIZE = int(0.8*NUM_EXAMPLES)
        BATCH_SIZE = 32
        BUFFER_SIZE = 1000
        STEPS_PER_EPOCH = NUM_EXAMPLES // BATCH_SIZE

    model_path = "pretrained/carvana_model.h5"
    test_img_path = "test_images/29bb3ece3180_11.jpg" 
    
    if opt.display_check:
        sample_image, sample_mask = imageio.imread(test_img_path), imageio.imread(test_img_path)
        display([sample_image, sample_mask])

    if opt.train_check:
        train_dataset, test_dataset = get_train_test_data(train_img_paths, train_mask_paths, TRAIN_SIZE, BATCH_SIZE)
        history, unet = train(train_dataset, test_dataset, opt.IMG_COLS, opt.IMG_ROWS, opt.save_model)
        if opt.display_check:
            plot_model_results(history)

    if opt.predict_check:
        if not opt.train_check:
            unet = load_model(model_path)

        img = np.array([cv2.resize(imageio.imread(test_img_path), (opt.IMG_ROWS, opt.IMG_COLS))])/255.0
        pred_mask = unet.predict(img)[0]
        bin_mask = 255. * cv2.resize(pred_mask, (opt.TEST_IMG_ROWS, opt.TEST_IMG_COLS))
        display([img, bin_mask])
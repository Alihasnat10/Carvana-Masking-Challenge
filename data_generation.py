import tensorflow as tf
from glob import glob
import zipfile

def extract_data(train_imgs_pth="../input/carvana-image-masking-challenge/train.zip", train_masks_pth="../input/carvana-image-masking-challenge/train_masks.zip", extract_to="../output/kaggle/working"):
    path_to_zip_file = train_imgs_pth
    directory_to_extract_to = "../output/kaggle/working"
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
        
    path_to_zip_file = train_masks_pth
    directory_to_extract_to = extract_to
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
    return "Done"
def get_data_paths(train_img_pth="../output/kaggle/working/train/*.jpg", train_mask_pth='../output/kaggle/working/train_masks/*.gif'):
    train_img_paths = sorted(glob(train_img_pth))
    train_mask_paths = sorted(glob(train_mask_pth))
    return train_img_paths, train_mask_paths

def normalize(input_image, input_mask):
    input_image = input_image / 255.0
    
    input_mask = tf.where(input_mask <= 127, 0., 1.)
    input_mask = tf.squeeze(input_mask, axis=-1)
    return input_image, input_mask

# Define a function to load and preprocess images and masks
def load_image_and_mask(image_path, mask_path):
    IMG_ROWS = 480
    IMG_COLS = 320
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMG_COLS, IMG_ROWS))
    image = tf.cast(image, tf.float32)
    
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_gif(mask)[0][:, :, 0]
    mask = tf.cast(mask, tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    mask = tf.image.resize(mask, size=(IMG_COLS, IMG_ROWS), method='nearest')
    image, mask = normalize(image, mask)
    return image, mask

def get_train_test_data(train_img_paths, train_mask_paths, TRAIN_SIZE, BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((train_img_paths, train_mask_paths))

    dataset = dataset.map(load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = dataset.take(TRAIN_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    test_dataset = dataset.skip(TRAIN_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, test_dataset

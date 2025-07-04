import tensorflow as tf
import os
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm

IMAGE_SIZE = (256, 256) 
# TRAIN_TEST_SPLIT = 0.8
TRAIN_SPLIT = 0.7 
TEST_SPLIT = 0.15 
VAL_SPLIT = 0.15  
DATA_DIR = "images_ori" 
OUTPUT_DIR = "tfrcds_ori"

os.makedirs(OUTPUT_DIR, exist_ok=True) #making sure 

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)
    return image

def write_tfrecord(file_paths, labels, output_file, pbar=None):
    with tf.io.TFRecordWriter(output_file) as writer:
        for image_path, label in zip(file_paths, labels):
            # Load and process the image
            image = load_image(image_path).numpy()
            
            # Convert image to uint8 if necessary
            image = (image * 255).astype("uint8")  # Rescale from [0, 1] to [0, 255]
            
            # Convert label to int64
            label = tf.convert_to_tensor(label, dtype=tf.int64).numpy()

            # Create a TFRecord example
            feature = {
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(image).numpy()])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            
            # Update progress bar
            if pbar:
                pbar.update(1)


image_paths = []
labels = []
class_names = sorted(os.listdir(DATA_DIR))
for label, class_name in enumerate(class_names):
  class_dir = os.path.join(DATA_DIR, class_name)
  class_images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith('.jpg')]
  image_paths.extend(class_images)
  labels.extend([label] * len(class_images))

# Shuffle data
combined = list(zip(image_paths, labels))
random.shuffle(combined)
image_paths, labels = zip(*combined)

# Step 1: Split into train and remaining (test + validation)
train_paths, remaining_paths, train_labels, remaining_labels = train_test_split(
    image_paths, labels, train_size=TRAIN_SPLIT, stratify=labels, random_state=42
)

# Step 2: Split remaining (test + validation) into test and validation sets
val_paths, test_paths, val_labels, test_labels = train_test_split(
    remaining_paths, remaining_labels, train_size=VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT),
    stratify=remaining_labels, random_state=42
)

# Print dataset sizes for verification
print(f"Training set size: {len(train_paths)}")
print(f"Validation set size: {len(val_paths)}")
print(f"Test set size: {len(test_paths)}")

# Define the output file paths
train_tfrecord_path = os.path.join(OUTPUT_DIR, "train.tfrecord")
test_tfrecord_path = os.path.join(OUTPUT_DIR, "test.tfrecord")
val_tfrecord_path = os.path.join(OUTPUT_DIR, "val.tfrecord")

# Write TFRecords for the training set with tqdm progress bar
print("Writing TFRecords for the training set...")
with tqdm(total=len(train_paths), desc="Training TFRecord", unit="image") as pbar:
    write_tfrecord(train_paths, train_labels, train_tfrecord_path, pbar)

# Write TFRecords for the test set with tqdm progress bar
print("Writing TFRecords for the test set...")
with tqdm(total=len(test_paths), desc="Test TFRecord", unit="image") as pbar:
    write_tfrecord(test_paths, test_labels, test_tfrecord_path, pbar)

# Write TFRecords for the validation set with tqdm progress bar
print("Writing TFRecords for the validation set...")
with tqdm(total=len(val_paths), desc="Validation TFRecord", unit="image") as pbar:
    write_tfrecord(val_paths, val_labels, val_tfrecord_path, pbar)

print(f"TFRecords saved to {OUTPUT_DIR}")


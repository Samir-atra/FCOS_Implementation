import os

import tensorflow as tf
from src.model.model import FCOS
from src.loss import IOULoss

# TPU Detection and Initialization
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU:", tpu.master())
except ValueError:
    print("TPU not found. Falling back to default strategy (CPU/GPU).")
    strategy = tf.distribute.get_strategy()

print("Replica count:", strategy.num_replicas_in_sync)

# Check for Kaggle Dataset Input
# Expected structure: /kaggle/input/coco2014
KAGGLE_DATASET_PATH = "/kaggle/input/coco2014"

# Fallback or alternate name
if not os.path.exists(KAGGLE_DATASET_PATH):
     KAGGLE_DATASET_PATH = "/kaggle/input/coco-2014-downloader"

if os.path.exists(KAGGLE_DATASET_PATH):
    print(f"Found dataset at {KAGGLE_DATASET_PATH}")
    # Check for internal structure (sometimes it's coco2014/coco2014 or just train2014)
    if os.path.exists(os.path.join(KAGGLE_DATASET_PATH, "train2014")):
        TRAIN_IMGS_PATH = os.path.join(KAGGLE_DATASET_PATH, "train2014")
        VAL_IMGS_PATH = os.path.join(KAGGLE_DATASET_PATH, "val2014")
        ANNOTATIONS_PATH = os.path.join(KAGGLE_DATASET_PATH, "annotations/instances_train2014.json")
    elif os.path.exists(os.path.join(KAGGLE_DATASET_PATH, "coco2014/train2014")):
         TRAIN_IMGS_PATH = os.path.join(KAGGLE_DATASET_PATH, "coco2014/train2014")
         VAL_IMGS_PATH = os.path.join(KAGGLE_DATASET_PATH, "coco2014/val2014")
         ANNOTATIONS_PATH = os.path.join(KAGGLE_DATASET_PATH, "coco2014/annotations/instances_train2014.json")
    else:
        # Fallback to direct path assuming simple structure
        TRAIN_IMGS_PATH = os.path.join(KAGGLE_DATASET_PATH, "train2014")
        VAL_IMGS_PATH = os.path.join(KAGGLE_DATASET_PATH, "val2014")
        ANNOTATIONS_PATH = os.path.join(KAGGLE_DATASET_PATH, "annotations/instances_train2014.json")

else:
    # Download COCO 2014 Dataset (if not present locally)
    if not os.path.exists("train2014.zip") and not os.path.exists("train2014"):
        print("Downloading COCO 2014 Dataset...")
        os.system("wget -q http://images.cocodataset.org/zips/train2014.zip")
        os.system("wget -q http://images.cocodataset.org/annotations/annotations_trainval2014.zip")
        print("Unzipping...")
        os.system("unzip -q train2014.zip")
        os.system("unzip -q annotations_trainval2014.zip")
        print("Data setup complete.")

    # Paths (Adjusted for Downloaded/Local Data)
    TRAIN_IMGS_PATH = "train2014/"
    VAL_IMGS_PATH = "val2014/" 
    ANNOTATIONS_PATH = "annotations/instances_train2014.json"

# Hyperparameters
BATCH_SIZE = 16 * strategy.num_replicas_in_sync  # Global batch size
EPOCHS = 12  # Standard for 1x schedule, adjust as needed

def create_learning_rate_schedule():
    # 1x Schedule: divide by 10 at 60k and 80k steps (approx epochs 8 and 11)
    # This needs to be converted to steps based on dataset size
    # For now, using a simple PiecewiseConstantDecay as placeholder
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[60000, 80000],
        values=[0.01, 0.001, 0.0001]
    )

from src.Data import load_data

with strategy.scope():
    # Model creation must happen inside the strategy scope
    model = FCOS()
    
    # Optimizer
    lr_schedule = create_learning_rate_schedule()
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=lr_schedule,
        momentum=0.9,
        weight_decay=0.0001
    )
    
    # Loss Definitiobs
    focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0)
    iou_loss = IOULoss() # Ensure this custom loss is compatible with distributed training
    bce_loss = tf.keras.losses.BinaryCrossentropy()
    
    # Compilation
    model.compile(
        optimizer=optimizer,
        loss={
            'classifier': focal_loss, 
            'box': iou_loss, 
            'centerness': bce_loss
        },
        metrics=['precision'] # Note: standard precision might not work well for OD, usually need COCOEvaluator
    )

def get_dataset(batch_size):
    return load_data.get_training_dataset(TRAIN_IMGS_PATH, ANNOTATIONS_PATH, batch_size)

# Training Loop
train_dataset = get_dataset(BATCH_SIZE)
if train_dataset:
     model.fit(
         train_dataset,
         epochs=EPOCHS
     )
print("Model initialized and compiled. Ready for Data Pipeline.")
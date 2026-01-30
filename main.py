import tensorflow as tf
import os
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

# Paths (Adjusted for Kaggle/Cloud environment - User should verify these)
TRAIN_IMGS_PATH = "/kaggle/input/coco-2017-dataset/coco2017/train2017/"
VAL_IMGS_PATH = "/home/samer/Desktop/Beedoo/FCOS/FCOS_Implementation/COCO2014/val2014/" # Placeholder
ANNOTATIONS_PATH = "/kaggle/input/coco-2017-dataset/coco2017/annotations/instances_train2017.json"

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
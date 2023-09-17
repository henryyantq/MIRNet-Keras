import random
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from mirnet import mirnet_model
from preprocess import get_dataset
from keras import optimizers, callbacks

random.seed(10)


DATASET_DIR = "./datasets/lol_dataset"

MAX_TRAIN_IMAGES = int(0.8*len(os.listdir(f"{DATASET_DIR}/our485/high")))
NUM_EPOCH = 200


# Define the Charbonnier loss function
def charbonnier_loss(y_true, y_pred):
    return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))


# Define the PSNR metric
def peak_signal_noise_ratio(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)


if __name__ == "__main__":

    train_low_light_images = sorted(glob(f"{DATASET_DIR}/our485/low/*"))[:MAX_TRAIN_IMAGES]
    train_enhanced_images = sorted(glob(f"{DATASET_DIR}/our485/high/*"))[:MAX_TRAIN_IMAGES]

    val_low_light_images = sorted(glob(f"{DATASET_DIR}/our485/low/*"))[MAX_TRAIN_IMAGES:]
    val_enhanced_images = sorted(glob(f"{DATASET_DIR}/our485/high/*"))[MAX_TRAIN_IMAGES:]

    test_low_light_images = sorted(glob(f"{DATASET_DIR}/eval15/low/*"))
    test_enhanced_images = sorted(glob(f"{DATASET_DIR}/eval15/high/*"))

    train_dataset = get_dataset(train_low_light_images, train_enhanced_images)
    val_dataset = get_dataset(val_low_light_images, val_enhanced_images)

    model = mirnet_model(num_rrg=3, num_mrb=2, channels=64)

    optimizer = optimizers.Adam(learning_rate=1e-4)

    model.compile(
        optimizer=optimizer, 
        loss=charbonnier_loss, 
        metrics=[peak_signal_noise_ratio],
    )

    checkpoint_saver = callbacks.ModelCheckpoint(
        "./checkpoints/mirnet-best_PSNR_{val_peak_signal_noise_ratio:.2f}/",
        monitor="val_peak_signal_noise_ratio",
        save_best_only=True,
        mode="max",
    )

    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor="val_peak_signal_noise_ratio",
        factor=0.5,
        patience=10,
        verbose=1,
        min_delta=1e-7,
        mode="max",
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=NUM_EPOCH,
        callbacks=[lr_scheduler, checkpoint_saver],
    )

    '''
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Losses Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(history.history["peak_signal_noise_ratio"], label="train_psnr")
    plt.plot(history.history["val_peak_signal_noise_ratio"], label="val_psnr")
    plt.xlabel("Epochs")
    plt.ylabel("PSNR")
    plt.title("Train and Validation PSNR Over Epochs", fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

    '''
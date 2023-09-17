import os
import numpy as np
from PIL import Image
from keras import utils
from keras.models import load_model
from mirnet import *
from train import charbonnier_loss, peak_signal_noise_ratio
    

MODEL_DIR = "./checkpoints"


def infer(model, original_image):
    image = utils.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output = model.predict(image)
    output_image = output[0] * 255.0
    output_image = output_image.clip(0, 255)
    output_image = output_image.reshape(
        (np.shape(output_image)[0], np.shape(output_image)[1], 3)
    )
    output_image = Image.fromarray(np.uint8(output_image))
    original_image = Image.fromarray(np.uint8(original_image))
    return output_image


if __name__ == "__main__":
    
    model_path = os.listdir(MODEL_DIR)[0]
    model_path = f"{MODEL_DIR}/{model_path}"

    model = load_model(
        model_path, custom_objects={
            "selective_kernel_feature_fusion": selective_kernel_feature_fusion,
            "spatial_attention_block": spatial_attention_block, 
            "channel_attention_block": channel_attention_block, 
            "dual_attention_unit_block": dual_attention_unit_block, 
            "down_sampling_module": down_sampling_module, 
            "up_sampling_module": up_sampling_module, 
            "multi_scale_residual_block": multi_scale_residual_block, 
            "recursive_residual_group": recursive_residual_group,
            "charbonnier_loss": charbonnier_loss,
            "peak_signal_noise_ratio": peak_signal_noise_ratio,
        },
    )

    print(f"\nModel loaded: {model_path}\n")

    while True:
        image_path = input("Please enter an image path >> ")
        
        if image_path == "exit" or image_path == "stop" or image_path == "break":
            print("\n[Exiting the programme...]\n")
            break

        save_path = input("Please enter a path for saving the result >> ")

        original_image = Image.open(image_path).convert("RGB")

        width, height = original_image.size

        while True:
            if width % 8 != 0:
                width -= 1
            else:
                break

        while True:
            if height % 8 != 0:
                height -= 1
            else:
                break
        
        original_image = original_image.crop((0, 0, width, height))

        print(original_image.size)

        print("\n[Infering...]\n")
        result = infer(model, original_image)

        result.save(save_path)
        print(f"\nDone. The result was saved to {save_path}.\n")


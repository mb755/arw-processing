import rawpy
import numpy as np
from skimage import exposure, restoration
import cv2
import os
import configparser as cfg

from utils.config_parser import default_parser

root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

###########################################################
# parse command line arguments
###########################################################

parser = default_parser(description="Save some marketdata to disk.")

args = vars(parser.parse_args())

input_file = args["input_file"]
output_suffix = args["output_suffix"]
config_file = args["config_file"]
overwrite = args["overwrite"]

###########################################################
# grab initial values from config file
###########################################################

config = cfg.ConfigParser()
config.read(config_file)

camera_name = config.get("camera", "name")
lens_name = config.get("lens", "name")


def process_raw(file_path):
    # 1. Read and apply basic corrections
    with rawpy.imread(file_path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True, half_size=False, no_auto_bright=False, output_bps=16
        )

    # Convert to float for further processing
    image = rgb.astype(np.float32) / 65535.0

    # 2. Auto adjustments
    # Auto-contrast
    p2, p98 = np.percentile(image, (2, 98))
    image = exposure.rescale_intensity(image, in_range=(p2, p98))

    # 3. Style/curve adjustments
    # Example: Increase contrast
    image = exposure.adjust_gamma(image, 1.2)

    # Increase saturation
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * 1.2  # Increase saturation by 20%
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    print(f"Image shape: {image.shape}", flush=True)

    # 4. Denoising and sharpening
    # Denoise
    image = restoration.denoise_wavelet(image)

    # Sharpen
    image = np.clip(image, 0, 1)  # Ensure values are in [0,1] range
    image = cv2.convertScaleAbs(image, alpha=255)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    image = cv2.filter2D(image, -1, sharpen_kernel)

    return image


# Usage
output = process_raw(input_file)
base_filename = os.path.basename(input_file).split(".")[0]
output_file = f"{root_dir}/output/{base_filename}{output_suffix}.jpg"
cv2.imwrite(output_file, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))

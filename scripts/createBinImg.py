import numpy as np
from PIL import Image

SRC_IMG = "../data/lena.png"
BIN_OUT = "../data/lena.bin"

dtype = np.uint8
image = Image.open(SRC_IMG)
image_np = np.asarray(image, dtype=dtype)

with open(BIN_OUT, "wb") as f:
    f.write(image_np.tobytes())
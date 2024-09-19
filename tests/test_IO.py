import pytest
import numpy as np
from PIL import Image
import gzip
import os
import io 

TOLERANCE = 1e-9
JPEG_QUALITY = 97

def load_img_from_gzip(bin_gz_path, shape, dtype=np.float64):
    with gzip.open(bin_gz_path, "rb") as f:
        data = f.read()
    imgMat = np.frombuffer(data, dtype=dtype).reshape(shape)
    
    return imgMat

def load_img_from_results(results_dir, img_file_res, dtype):
    img = Image.open(os.path.join(results_dir, img_file_res))
    return np.asarray(img, dtype=dtype)

@pytest.fixture(scope='session')
def images_dir():
    return "../data/"

@pytest.fixture(scope='session')
def results_dir():
    return "../results/"

@pytest.fixture(scope='function')
def load_img(request, images_dir, results_dir):
    img_file, shape, dtype, img_file_res = request.param
    img_path = os.path.join(images_dir, img_file)
    
    imgMatDT = load_img_from_gzip(img_path, shape, dtype)
    imgMatCalc = load_img_from_results(results_dir, img_file_res, dtype)
    
    if img_file == "lena_hd_t.bin.gz":
        imgMatDT = np.transpose(imgMatDT, (1,0,2))
    
    if img_file_res.endswith(".jpg"):
        img = Image.fromarray(imgMatDT)
        buffer = io.BytesIO()
        img.save(buffer, format = "JPEG", quality = JPEG_QUALITY)
        img = Image.open(buffer)
        imgMatDT = np.asarray(img, dtype=dtype)
    
    return imgMatDT, imgMatCalc

@pytest.mark.parametrize("load_img", [
                                        ("elvis.bin.gz", (469,700), np.float64, "elvis.png"),
                                        ("lena_hd.bin.gz", (822,1200,3), np.uint8, "lena_hd.jpg"),
                                        ("lena_hd2.bin.gz", (1960,1960,3), np.uint8, "lena_hd2.jpg"),
                                        ("lena_hd_t.bin.gz", (1200,822,3), np.uint8, "lena_hd_t.png"),
                                        ], indirect=True)
def test_load_img(load_img):
    imgDT, imgCalc = load_img
    
    assert imgCalc.shape == imgDT.shape
    assert imgCalc.dtype == imgDT.dtype
    assert np.allclose(imgCalc, imgDT, atol=TOLERANCE)

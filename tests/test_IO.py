import pytest
import numpy as np
from PIL import Image

def load_img_from_gzip(bin_gz_path, shape, dtype = np.float64):
    with gzip.open(bin_gz_path, "rb") as f:
        data = f.read()
    imgMat = np.frombuffer(data,dtype=dtype).reshape(shape)
    return imgMat

def load_img_from_results(results_dir, img_file_res):
    img = Image.open(os.path.join(results_dir, img_file_res))
    return np.asarray(img)

@pytest.fixture(scope = 'session')
def images_dir():
    return "../data/"
@pytest.fixture(scope = 'session')
def results_dir():
    return "../results/"

@pytest.fixture(scope = 'function')
def load_img(req, images_dir, results_dir):
    img_file, shape, dtype, img_file_res = req.param
    img_path = os.path.join(images_dir, img_file)
    
    imgMatDT = load_img_from_gzip(img_path, shape, dtype)
    imgMatCalc = load_img_from_results(results_dir, img_file_res)
    return imgMatDT, imgMatCalc

@pytest.mark.parametrize("load_img", [
                                        ("elvis.bin.gz", (469,700), np.float64, "elvis.png"),
                                        # ("cyclone.bin.gz", (4096,4096), np.float64, ""),
                                        # ("earth.bin.gz", (9500,9500), np.float64),
                                        # ("lena.bin.gz", (512,512,3), np.uint8),
                                        ("lena_hd.bin.gz", (822,1200,3), np.uint8, "lena_hd.bin"),
                                        ("lena_hd.bin.gz", (822,1200,3), np.uint8, "lena_hd.png"),
                                        ("lena_hd.bin.gz", (822,1200,3), np.uint8, "lena_hd.jpg"),
                                        ("lena_hd2.bin.gz", (1960,1960,3), np.uint8, "lena_hd2.jpg"),
                                        ("lena_hd_t.bin.gz", (822,1200,3), np.uint8, "lena_hd_t.png"),
                                        ], indirect=True)
def test_load_img(load_img):
    
    imgDT, imgCalc = load_img
    
    
    assert imgCalc.shape == ImgDT.shape
    assert imgCalc.dtype == imgDT.dtype
    assert np.allclose(imgCalc, imgDT, atol=1e-5)
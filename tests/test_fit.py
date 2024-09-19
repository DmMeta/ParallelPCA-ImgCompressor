import pytest
import numpy as np
import gzip
import os

TOLERANCE = 1e-8

def load_img(bin_gz_path, shape, dtype=np.float64):
    with gzip.open(bin_gz_path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=dtype).reshape(shape)

def load_stats_from_results(mean_file, std_file, dtype = np.float64):
    mean = np.loadtxt(mean_file, dtype=dtype, delimiter=',')
    std = np.loadtxt(std_file, dtype=dtype, delimiter=',')
    return mean, std

def compute_stats(img_path, shape, dtype):
    imgMat = load_img(img_path, shape, dtype)
    return np.mean(imgMat, axis=0), np.std(imgMat, axis=0)

@pytest.fixture(scope='session')
def images_dir():
    return "../data/"

@pytest.fixture(scope='session')
def stats_files():
    return "../results/mean", "../results/std"

@pytest.fixture(scope='function')
def load_stats(request, images_dir, stats_files):
    img_file, shape, dtype = request.param
    img_path = os.path.join(images_dir, img_file)
    
    npMean, npStd = compute_stats(img_path, shape, dtype)
    mean_file, std_file = stats_files
    calcMean, calcStd = load_stats_from_results(mean_file, std_file, dtype = np.float64)
    return (npMean, npStd), (calcMean, calcStd)

@pytest.mark.parametrize("load_stats", [("lena_hd.bin.gz", (822,1200,3), np.uint8)], indirect=True)
def test_stats(load_stats):
    (npMean, npStd), (calcMean, calcStd) = load_stats
    
    if npMean.shape != calcMean.shape:
        npMean = npMean.T
        npStd = npStd.T

    assert np.allclose(calcMean, npMean, atol=TOLERANCE)
    assert np.allclose(calcStd, npStd, atol=TOLERANCE)

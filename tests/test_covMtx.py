import pytest
import numpy as np
import gzip
import os

TOLERANCE = 1e-8

def load_img(bin_gz_path, shape, dtype=np.float64):
    with gzip.open(bin_gz_path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=dtype).reshape(shape)

def load_covMtx_from_results(covMtx_file, dtype = np.float64):
    covMtx = np.loadtxt(covMtx_file, dtype=dtype, delimiter=',')
    return covMtx

def compute_stats(img):
    return np.mean(img, axis=0), np.std(img, axis=0)

def compute_covMtx(img_path, shape, dtype):
    imgMat = load_img(img_path, shape, dtype)
    mean, std = compute_stats(imgMat)
    imgMatNorm = (imgMat - mean) / std
    if (imgMatNorm.ndim > 2):
        covMtx_comb = []
        for ch in range(imgMatNorm.shape[2]):
            channelCovMtx = np.cov(imgMatNorm[:, :, ch].T) # channelCovMtx.shape = (imgMat.shape[1], imgMat.shape[1])
            covMtx_comb.append(channelCovMtx)
        # covMtx_comb has channels elements, each element is a covariance matrix of shape (imgMat.shape[1], imgMat.shape[1]) corresponding to a channel
        
        covMtx = np.vstack(covMtx_comb) # covMtx.shape = (channels * imgMat.shape[1], imgMat.shape[1])
    else:
        covMtx = np.cov(imgMatNorm.T)
            
    return covMtx

@pytest.fixture(scope='session')
def images_dir():
    return "../data/"

@pytest.fixture(scope='session')
def results_dir():
    return "../results/covarianceMatrix"

@pytest.fixture(scope='function')
def load_covMtx(request, images_dir, results_dir):
    img_file, shape, dtype = request.param
    img_path = os.path.join(images_dir, img_file)
    
    npCovMtx = compute_covMtx(img_path, shape, dtype)
    calcCovMtx = load_covMtx_from_results(results_dir, dtype = np.float64)
    return npCovMtx, calcCovMtx

@pytest.mark.parametrize("load_covMtx", [("elvis.bin.gz", (469, 700), np.float64)], indirect=True)
def test_covMtx(load_covMtx):
    npCovMtx, calcCovMtx = load_covMtx
    
    assert np.allclose(calcCovMtx, npCovMtx, atol=TOLERANCE)
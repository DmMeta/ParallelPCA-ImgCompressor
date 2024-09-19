import pytest
import numpy as np
import gzip
import os

TOLERANCE = 1e-8

def load_img(bin_gz_path, shape, dtype=np.float64):
    with gzip.open(bin_gz_path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=dtype).reshape(shape)

def load_eigv_from_results(eigv_file, dtype = np.float64):
    eigv = np.loadtxt(eigv_file, dtype=dtype, delimiter=',')
    return eigv

def compute_stats(img):
    return np.mean(img, axis=0), np.std(img, axis=0)

def compute_covMtx(imgMat):
    mean, std = compute_stats(imgMat)
    imgMatNorm = (imgMat - mean) / std
    if (imgMatNorm.ndim > 2):
        covMtx_comb = []
        for ch in range(imgMatNorm.shape[2]):
            channelCovMtx = np.cov(imgMatNorm[:, :, ch].T)
            covMtx_comb.append(channelCovMtx)
        covMtx = np.vstack(covMtx_comb) # covMtx.shape: (channels * imgMat.shape[1], imgMat.shape[1])
    else:
        covMtx = np.cov(imgMatNorm.T)
            
    return covMtx

def compute_eigv(img_path, shape, dtype, princ_comp):
    imgMat = load_img(img_path, shape, dtype)
    assert princ_comp <= imgMat.shape[1]
    covMtx = compute_covMtx(imgMat)
    
    if (imgMat.ndim > 2):
        channels = imgMat.shape[2]
        rows = covMtx.shape[0] // channels 
        eigenvComp = []
        for ch in range(channels):
            _, eigv = np.linalg.eigh(covMtx[ch * rows:(ch + 1) * rows, :]) # eigv.shape: (imgMat.shape[1], imgMat.shape[1])
            eigenvComp.append(eigv[:, -princ_comp:])
        # eigenvComp has channels elements -> each element is a matrix of shape (imgMat.shape[1], princ_comp) corresponding to a channel
        eigv = np.hstack(eigenvComp) # eigv.shape: (imgMat.shape[1], channels * princ_comp)
    else:
        _, eigv = np.linalg.eigh(covMtx)
        eigv = eigv[:, -princ_comp:]
            
    return eigv.T # eigv.T.shape: (channels * princ_comp, imgMat.shape[1])

@pytest.fixture(scope='session')
def images_dir():
    return "../data/"

@pytest.fixture(scope='session')
def eigenvectors_file():
    return "../results/eigenVectors"

@pytest.fixture(scope='function')
def load_eigv(request, images_dir, eigenvectors_file):
    img_file, shape, dtype, princ_comp = request.param
    img_path = os.path.join(images_dir, img_file)
    
    npEigv = compute_eigv(img_path, shape, dtype, princ_comp)
    calcEigv = load_eigv_from_results(eigenvectors_file, dtype = np.float64)
    
    return npEigv, calcEigv

@pytest.mark.parametrize("load_eigv", [("lena_hd.bin.gz", (822, 1200, 3), np.uint8, 100)], indirect=True)
def test_eigv(load_eigv):
    npEigv, calcEigv = load_eigv
    
    # We compare the absolute values of the eigenvectors since their signs are arbitrary 
    # and can differ between the two eigenvector computation methods.
    assert np.allclose(np.abs(calcEigv), np.abs(npEigv), atol=TOLERANCE)
import pytest
import numpy as np
import gzip
import os

TOLERANCE = 1e-8

def load_img(bin_gz_path, shape, dtype=np.float64):
    with gzip.open(bin_gz_path, "rb") as f:
        data = f.read()
    return np.frombuffer(data, dtype=dtype).reshape(shape)

def load_comprImg_from_results(comprImg_file, dtype = np.float64):
    comprImg = np.loadtxt(comprImg_file, dtype=dtype, delimiter=',')
    return comprImg

def compute_stats(img):
    return np.mean(img, axis=0), np.std(img, axis=0)

def compute_covMtx(imgMat):
    if (imgMat.ndim > 2):
        covMtx_comb = []
        for ch in range(imgMat.shape[2]):
            channelCovMtx = np.cov(imgMat[:, :, ch].T)
            covMtx_comb.append(channelCovMtx)
        covMtx = np.vstack(covMtx_comb) 
    else:
        covMtx = np.cov(imgMat.T)
            
    return covMtx

def compute_eigv(covMtx, channels, princ_comp):
    if (channels > 1):
        rows = covMtx.shape[0] // channels 
        eigenvComp = []
        for ch in range(channels):
            _, eigv = np.linalg.eigh(covMtx[ch * rows:(ch + 1) * rows, :])
            eigenvComp.append(eigv[:, -princ_comp:])
            
        eigv = np.hstack(eigenvComp)
    else:
        _, eigv = np.linalg.eigh(covMtx)
        eigv = eigv[:, -princ_comp:]
            
    return eigv 

def compute_comprImg(img_path, shape, dtype, princ_comp):
    imgMat = load_img(img_path, shape, dtype)
    assert princ_comp <= imgMat.shape[1]
    channels = imgMat.shape[2] if imgMat.ndim > 2 else 1
    mean, std = compute_stats(imgMat)
    imgMatNorm = (imgMat - mean) / std
    covMtx = compute_covMtx(imgMatNorm)
    eigv = compute_eigv(covMtx, channels, princ_comp)
   
    if (channels > 1):
        rows = eigv.shape[1] // channels
        comprImg_comb = []
        for ch in range(channels):
            # channelComprImg.shape: (imgMat.shape[0], princ_comp)
            channelComprImg = np.dot(imgMatNorm[:, :, ch], eigv[:, ch * rows:(ch + 1) * rows]) 
            comprImg_comb.append(channelComprImg)
        # comprImg_comb has channels elements -> each element is a matrix of shape (imgMat.shape[0], princ_comp) corresponding to a channel
        comprImg = np.vstack(comprImg_comb) # comprImg.shape: (imgMat.shape[0] * channels, princ_comp)
        
    else:
        comprImg = np.dot(imgMatNorm, eigv)
    
    return comprImg

@pytest.fixture(scope='session')
def images_dir():
    return "../data/"

@pytest.fixture(scope='session')
def results_dir():
    return "../results/projection"

@pytest.fixture(scope='function')
def load_comprImg(request, images_dir, results_dir):
    img_file, shape, dtype, princ_comp = request.param
    img_path = os.path.join(images_dir, img_file)
    
    npComprImg = compute_comprImg(img_path, shape, dtype, princ_comp)
    calcComprImg = load_comprImg_from_results(results_dir, dtype = np.float64)
    
    return npComprImg, calcComprImg

@pytest.mark.parametrize("load_comprImg", [("elvis.bin.gz", (469, 700), np.float64, 100)], indirect=True)
def test_projection(load_comprImg):
    npComprImg, calcComprImg = load_comprImg
    
    assert np.allclose(np.abs(calcComprImg), np.abs(npComprImg), atol=TOLERANCE)
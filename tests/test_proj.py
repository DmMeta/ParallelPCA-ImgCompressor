import pytest
import numpy as np
import gzip
from PIL import Image
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

def compute_comprImg(imgMat, shape, dtype, princ_comp):
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
    
    return comprImg, eigv

@pytest.fixture(scope='session')
def images_dir():
    return "../data/"

@pytest.fixture(scope='session')
def projection_file():
    return "../results/projection"

@pytest.fixture(scope='session')
def decompImg_file():
    return "../results/lena_hd_decompressed.png"

@pytest.fixture(scope='function')
def load_comprImg(request, images_dir, projection_file):
    img_file, shape, dtype, princ_comp = request.param
    img_path = os.path.join(images_dir, img_file)
    imgMat = load_img(img_path, shape, dtype)
    
    npComprImg, _ = compute_comprImg(imgMat, shape, dtype, princ_comp)
    calcComprImg = load_comprImg_from_results(projection_file, dtype = np.float64)
    
    return npComprImg, calcComprImg

@pytest.fixture(scope='function')
def load_decompImg(request, images_dir, projection_file):
    img_file, shape, dtype, princ_comp = request.param
    img_path = os.path.join(images_dir, img_file)
    imgMat = load_img(img_path, shape, dtype)
    
    npComprImg, principal_comp = compute_comprImg(imgMat, shape, dtype, princ_comp)
   
    principal_comp = principal_comp.T
    try:
        channels = shape[2]
    except IndexError:
        channels = 1
        
    if (channels > 1):
        rows = principal_comp.shape[0] // channels
        
        compressedImgCollection = [npComprImg[ch * imgMat.shape[0]: (ch + 1) * imgMat.shape[0], :] for ch in range(channels)]
        npComprImg = np.stack(compressedImgCollection, axis = 2)
  
        
        decompImg_comb = []
        for ch in range(channels):
            # channelComprImg.shape: (imgMat.shape[0], princ_comp)
            channelComprImg = np.dot(npComprImg[:, :, ch], principal_comp[ch * rows:(ch + 1) * rows, :]) 
            decompImg_comb.append(channelComprImg)
        # comprImg_comb has channels elements -> each element is a matrix of shape (imgMat.shape[0], princ_comp) corresponding to a channel
        decomp_shape = decompImg_comb[0].shape
        decompImg = np.stack(decompImg_comb, axis = 2)
       
    else:
        decompImg = np.dot(npComprImg, principal_comp)
    
    mean, std = compute_stats(imgMat)
    decompImg = decompImg * std + mean
    
    
    return np.rint(decompImg)

@pytest.mark.parametrize("load_comprImg", [("lena_hd.bin.gz", (822, 1200, 3), np.uint8, 100)], indirect=True)
def test_projection(load_comprImg):
    npComprImg, calcComprImg = load_comprImg
    
    assert np.allclose(np.abs(calcComprImg), np.abs(npComprImg), atol=TOLERANCE)


@pytest.mark.parametrize("load_decompImg", [("lena_hd.bin.gz", (822, 1200, 3), np.uint8, 100)], indirect=True)
def test_invprojection(load_decompImg, decompImg_file):
    npDecompImg = np.clip(load_decompImg, 0, 255)
    
    dtype = npDecompImg.dtype
    img = Image.open(decompImg_file, 'r')
    calcDecompImg = np.asarray(img, dtype = dtype)
 
    # num_differences = np.transpose((npDecompImg != calcDecompImg).nonzero())
    
    
    assert np.allclose(calcDecompImg, npDecompImg, atol=TOLERANCE)
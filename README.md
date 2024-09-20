<h1 align="center">
<strong>Image Compression: A PCA-based approach enhanced by parallel processing</strong>
</h1>


<h4 align="center">
    <img src="https://img.shields.io/badge/License-MIT-%2300599C.svg" alt="MIT" style="height: 20px;">
    <img src="https://img.shields.io/badge/C++-17-%2300599C.svg?logo=c%2B%2B&logoColor=white" alt="cpp" style="height: 20px;">
    <img src="https://img.shields.io/badge/Python-3.10-%2300599C.svg?logo=python&logoColor=white" alt="python" style="height: 20px;">
 
</h4>




## 🚩 Table of Contents
* [Introduction](#introduction)
* [Key Features](#key-features)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Usage](#usage)
* [Tests](#tests)
* [Contact](#contact)
* [License](#license)


## Introduction
In this project we implement the standard Covariance method for the PCA procedure in **modern C++17**, applying it to both grayscale and RGB image compression. The computational backbone of this implementation harnesses the power of [OpenBlas][openblas-link] and [Lapacke][lapacke-link] in order to perform high-performance matrix operations and linear algebraic computations. Additionally, to further enhance performance, we utilize parallel processing through multithreading with [OpenMP][openmp-link], alongside cache memory alignment techniques and vectorization (**SIMD**) via CPU intrinsics. 

## Key Features
- Current version supports only loading images stored as binary gzipped files.
- Converting and storing compressed images as binary, PNG, JPEG or gzip.
- Fully functional PCA algorithm for grayscale and RGB images.
- The primary procedure focuses on computing the eigenvalues of the covariance matrix, as detailed [here][method-link].
- Implemented flattened storage of input image to align with cache line.
- Parallel multithreaded version of fit and transform of input image (Z-normalization).
- Parallel multithreaded and vectorized computation of the covariance matrix.
- Lapacke computation of eigenvectors with [dsyev()][dsyev-link].
- Parallel projection via matrix multiplication (utilizing CBlas [dgemm()][dgemm-link] for grayscale images). For the RGB case we parallelize the matrix multiplication using OpenMP.
- Implemented parallelized inverse PCA procedure (decompression), enabling efficient reconstruction of original image data.

>[!NOTE]
> `dsyev()` operates sequentially, which may lead to significant overhead, epsecially when processing larger images.

### TODOs
- Explore matrix-free methods for an alternative PCA procedure, eliminating the need for full eigenvector calculation and the construction and storing of the covariance matrix.
- Enable the loading of images from additional formats.


## Prerequisites
- Docker
- Python => 3.10.*
- A list of Python libraries listed [here][requirements-link]

## Installation
We provide a custom [Dockerfile][dockerfile-link] that manages all required dependencies, including GCC, OpenBlas and Lapacke.

```bash
git clone https://github.com/DmMeta/ParallelPCA-ImgCompressor
cd ParallelPCA-ImgCompressor
docker build -t pca:v0.4 .
``` 
>[!NOTE]
> Naming the image `pca` and tagging it as `v0.4` ensures consistency with the provided running scripts.

## Usage
By default, we provide a run [script][script-link] that compiles and executes the main driver program inside a container. You can alter the behavior of the driver program passing two environment variables `DEBUG` and `SIMD`. 
To run the script with *debug mode* and *SIMD optmizations* disabled:
```bash
DEBUG=0 SIMD=0 ./run.sh
``` 
The execution above triggers the image compression of the [lena_hd][lena-link] image, storing two images in a results folder, one before and one after the procedure takes place.
With `DEBUG=1` one should expect the runtimes of each part of the computation and the intermediate results as well as a series of images.

The following example illustrates the main functionalities of the current project:
```C++
#include "pca.hpp"

constexpr const uint16_t N_COMPONENTS = 20;

ImgMatrix<uint8_t> img {"path/to/image.bin.gz", height, width, channels, Order::ROW_MAJOR};
PCA<uint8_t> pca;
auto compressedImg = pca.performPCA(img, N_COMPONENTS);
auto decompressedImg = pca.inversePCA(compressedImg);

// save the decompressed image
decompressedImg.saveImg("path/to/decompressedImg.png", ImageFormat::PNG);
```


## Tests
Some initial tests have been implemented using the Pytest library; however, the current test coverage remains suboptimal. 
We aim to enhance our testing framework in the future to ensure better coverage and reliability of the code.

Install the required dependencies:
```bash
pip3 install -r requirements.txt
cd scripts
# ensure execute permissions are granted for run_tests.sh
chmod +x run_tests.sh
./run_tests.sh
```


## Contact
- Metaxakis Dimitris | <a href="mailto:d.metaxakis@ac.upatras.gr">d.metaxakis@ac.upatras.gr</a>
- Sofotasios Argiris | <a href="mailto:a.sofotasios@ac.upatras.gr">a.sofotasios@ac.upatras.gr</a>


## License
Distributed under the [MIT] License. See `LICENSE.md` for more details.

<!-- MARKDOWN LINKS -->
[openblas-link]:https://github.com/OpenMathLib/OpenBLAS
[lapacke-link]: https://www.netlib.org/lapack/lapacke.html
[requirements-link]: ./requirements.txt
[openmp-link]: https://www.openmp.org/
[dockerfile-link]: ./Dockerfile
[script-link]: ./scripts/run.sh
[lena-link]: ./data/lena_hd.bin.gz
[MIT]: https://en.wikipedia.org/wiki/MIT_License
[method-link]: https://visualstudiomagazine.com/Articles/2024/01/17/principal-component-analysis.aspx
[dsyev-link]: https://netlib.org/lapack/explore-html-3.6.1/d2/d8a/group__double_s_yeigen_ga442c43fca5493590f8f26cf42fed4044.html
[dgemm-link]: https://www.netlib.org/lapack/explore-html/dd/d09/group__gemm_ga1e899f8453bcbfde78e91a86a2dab984.html#ga1e899f8453bcbfde78e91a86a2dab984
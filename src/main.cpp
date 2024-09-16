#include <iostream>
// #include <cstdint>
#include "pca.hpp"

///////Dataset Info///////
// Elvis Image size: 469x700 (GrayScale double)
// Lena Image size: 512x512 (RGB uint8)
// Lena_hd Image size: 822x1200 (RGB uint8)
// Lena_hd_t Image size: 1200x822 (RGB uint8)
// Lena_hd2 Image size: 1960x1960 (RGB uint8)
// Cyclone Image size: 4096x4096 (GrayScale double)
// Earth Image size: 9500x9500 (GrayScale double)
//////////////////////////

constexpr const std::string_view results_path = "/opt/ImgCompression/results/";

int main() {
    
   
    ImgMatrix<double> img{"/data/elvis.bin.gz", 469,700, static_cast<uint8_t>(1), Order::ROW_MAJOR}; 
    img.saveImg("elvis.png", ImageFormat::PNG);
    
    // for(size_t i = 0; i < 5; i++){rows, cols)))
    //     std::cout << (img(i,i,0)) << std::endl;
    // }

    // img.saveImg("output_image.bin", ImageFormat::BIN);

    PCA<double> pca;
    auto compressedImg = pca.performPCA(img, 100);
    // std::cout << "Cache line size:" << sysconf (_SC_LEVEL1_DCACHE_LINESIZE) << std::endl;
    // compressedImg.saveImg("compressed_image.bin", ImageFormat::BIN);
    auto decompressedImg = pca.inversePCA(compressedImg);
    // decompressedImg.saveImg(std::string{results_path} + std::string{"elvis_decompressed"}, ImageFormat::PNG);
    
#ifdef _DEBUG_
ImgMatrix<double> img2{"/data/lena_hd.bin.gz", 822, 1200, static_cast<uint8_t>(3), Order::ROW_MAJOR}; 
img2.saveImg(std::string{results_path} + std::string{"lena_hd.bin"}, ImageFormat::BIN);
img2.saveImg(std::string{results_path} + std::string{"lena_hd.png"}, ImageFormat::PNG);
img2.saveImg(std::string{results_path} + std::string{"lena_hd.jpg"}, ImageFormat::JPG);
ImgMatrix<double> img3{"/data/lena_hd2.bin.gz", 1960, 1960, static_cast<uint8_t>(3), Order::ROW_MAJOR}; 
img3.saveImg(std::string{results_path} + std::string{"lena_hd2.bin"}, ImageFormat::JPG);
ImgMatrix<double> img4{"/data/lena_hd_t.bin.gz", 822, 1200 static_cast<uint8_t>(3), Order::COL_MAJOR}; 
img4.saveImg(std::string{results_path} + std::string{"lena_hd_t.bin"}, ImageFormat::PNG);
#endif



    return 0;
}



// cv::Mat image(height, width, CV_8UC1, img.data().data());
// cv::imwrite("output_image.png", image);
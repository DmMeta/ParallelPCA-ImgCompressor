#include <iostream>
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
    
   
    ImgMatrix<uint8_t> img{"/data/lena_hd.bin.gz", 822, 1200, static_cast<uint8_t>(3), Order::ROW_MAJOR}; 
    img.saveImg(std::string{results_path} + std::string{"lena_hd.png"}, ImageFormat::PNG);
    

    PCA<uint8_t> pca;
    auto compressedImg = pca.performPCA(img, 100);
    // std::cout << "Cache line size:" << sysconf (_SC_LEVEL1_DCACHE_LINESIZE) << std::endl;
    // compressedImg.saveImg("compressed_image.bin", ImageFormat::BIN);
    auto decompressedImg = pca.inversePCA(compressedImg);
    decompressedImg.saveImg(std::string{results_path} + std::string{"lena_hd_decompressed"}, ImageFormat::PNG);
    
#ifdef _DEBUG_
ImgMatrix<uint8_t> img2{"/data/lena_hd.bin.gz", 822, 1200, static_cast<uint8_t>(3), Order::ROW_MAJOR}; 
img2.saveImg(std::string{results_path} + std::string{"lena_hd"}, ImageFormat::PNG);
img2.saveImg(std::string{results_path} + std::string{"lena_hd"}, ImageFormat::JPG);
ImgMatrix<uint8_t> img3{"/data/lena_hd2.bin.gz", 1960, 1960, static_cast<uint8_t>(3), Order::ROW_MAJOR}; 
img3.saveImg(std::string{results_path} + std::string{"lena_hd2"}, ImageFormat::JPG);
ImgMatrix<uint8_t> img4{"/data/lena_hd_t.bin.gz", 822, 1200, static_cast<uint8_t>(3), Order::COLUMN_MAJOR}; 
img4.saveImg(std::string{results_path} + std::string{"lena_hd_t"}, ImageFormat::PNG);
#endif



    return 0;
}
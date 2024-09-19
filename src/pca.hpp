#pragma once

#include <string>
#include <tuple>
#include <vector>
#include <array>
#include <optional>
#include <cstdint>
#include <string_view>
#include <stdexcept>
#include <zlib.h>
#include <cstdint>
#include <iostream>
#include <utility>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <iomanip>
#include <lapacke.h>
#include <cblas.h>
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <limits>
#include <new>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <type_traits>


template<typename ElementType, std::size_t ALIGNMENT_IN_BYTES = 64>
class AlignedAllocator
{
private:
    static_assert(
        ALIGNMENT_IN_BYTES >= alignof( ElementType ),
        "Beware that types like int have minimum alignment requirements or access will result in crashes."
    );

public:
    using value_type = ElementType;
    static std::align_val_t constexpr ALIGNMENT{ ALIGNMENT_IN_BYTES };

   
    template<class OtherElementType>
    struct rebind
    {
        using other = AlignedAllocator<OtherElementType, ALIGNMENT_IN_BYTES>;
    };

public:
    constexpr AlignedAllocator() noexcept = default;

    constexpr AlignedAllocator( const AlignedAllocator& ) noexcept = default;

    template<typename U>
    constexpr AlignedAllocator( AlignedAllocator<U, ALIGNMENT_IN_BYTES> const& ) noexcept
    {}

    [[nodiscard]] ElementType*
    allocate( std::size_t nElementsToAllocate )
    {
        if ( nElementsToAllocate
             > std::numeric_limits<std::size_t>::max() / sizeof( ElementType ) ) {
            throw std::bad_array_new_length();
        }

        auto const nBytesToAllocate = nElementsToAllocate * sizeof( ElementType );
        return reinterpret_cast<ElementType*>(
            ::operator new[]( nBytesToAllocate, ALIGNMENT ) );
    }

    void
    deallocate(                  ElementType* allocatedPointer,
                [[maybe_unused]] std::size_t  nBytesAllocated )
    {
       
        ::operator delete[]( allocatedPointer, ALIGNMENT );
    }
};  

// check cache line size and change ALIGNMENT_IN_BYTES accordingly
template<typename T, std::size_t ALIGNMENT_IN_BYTES = 64>
using AlignedVec = std::vector<T, AlignedAllocator<T, ALIGNMENT_IN_BYTES>>;

enum class Order {
    ROW_MAJOR,
    COLUMN_MAJOR
};

enum class ImageFormat {
    BIN,
    BIN_GZIP,
    PNG,
    JPG
};

void print_times(std::array<int64_t, 5> times);

template<typename T>
struct is_2d_vector : std::false_type {};

template<typename T>
struct is_2d_vector<std::vector<std::vector<T>>> : std::true_type {};

template <typename T>
typename std::enable_if<!is_2d_vector<T>::value>::type 
reportResults(std::string_view filename, const T& data, std::tuple<uint16_t, uint16_t, uint8_t> shape) {

    std::filesystem::path path = "/opt/ImgCompression/results/";

    try {
        if(!std::filesystem::exists(path))
            std::filesystem::create_directory(path);

        path /= filename;
        std::ofstream file(path);
        
        if (!file.is_open())
            throw std::runtime_error("Error opening file for writing.");
            
        auto [rows, cols, channels] = shape;
        for (auto k = 0; k < channels; k++){                    
            for(auto i = 0; i < rows; i++){
                for(auto j = 0; j < cols; j++){
                    auto sep = std::string(j == cols - 1 ? "\n" : ",");
                    file << std::fixed << std::setprecision(16) << data[i*cols + j + k*rows*cols] << sep;
                }
            }
        }
        file.close();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

}

template<typename T>
typename std::enable_if<is_2d_vector<T>::value>::type
reportResults(std::string_view filename, const T& data, std::tuple<uint16_t, uint16_t, uint8_t> shape) {

    std::filesystem::path path = "/opt/ImgCompression/results/";

    try {
        if(!std::filesystem::exists(path))
            std::filesystem::create_directory(path);

        path /= filename;
        std::ofstream file(path);
        
        if (!file.is_open())
            throw std::runtime_error("Error opening file for writing.");
            
        auto [rows, cols, channels] = shape;

        for (auto k = 0; k < channels; k++){                    
            for(auto i = 0; i < rows; i++){
                for(auto j = 0; j < cols; j++){
                    auto sep = std::string(j == cols - 1 ? "\n" : ",");
                    file << std::fixed << std::setprecision(8) << data[k][i*cols + j]<< sep;
                }
            }
        }


        file.close();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

}

inline double reduce_sum(__m256d sum)
{
    double result[2];
    
    __m128d low = _mm256_castpd256_pd128(sum);
    __m128d high = _mm256_extractf128_pd(sum, 1);
    __m128d sum_ = _mm_add_pd(low, high);
    
    _mm_storeu_pd(result, sum_);
    
    return static_cast<double>(result[0] + result[1]);
}




constexpr const uint8_t img_quality = 97;

template<typename T>
class ImgMatrix {
    public:
        explicit ImgMatrix(std::string_view filename, 
                            uint16_t rows, uint16_t columns, uint8_t channels,
                            Order order = Order::ROW_MAJOR);
        ImgMatrix(uint16_t rows, uint16_t columns, uint8_t channels, 
                  Order order = Order::ROW_MAJOR);
        T& operator()(size_t i, size_t j, size_t k);
        const T& operator()(size_t i, size_t j, size_t k) const;
        bool saveImg(std::string_view outputFilename, ImageFormat fmt = ImageFormat::BIN);
        std::tuple<uint16_t, uint16_t, uint8_t> shape() const;
        AlignedVec<T>& data();
        const AlignedVec<T>& data() const;
        static AlignedVec<T> transpose(const AlignedVec<T>& data, std::tuple<uint16_t, uint16_t, uint8_t> shape_);
    
    private:
        std::string imgPath_;
        std::tuple<uint16_t, uint16_t, uint8_t> shape_;
        AlignedVec<T> imgData_;
        
        std::optional<AlignedVec<T>> readGzipFile(Order order = Order::ROW_MAJOR);
        void saveAsBinary(std::string_view filename);
        void saveAsCompressedBinary(std::string_view filename);
        void saveAsImg(std::string_view filename, ImageFormat fmt);
        void writeImg(std::string_view imgFilename, ImageFormat fmt, cv::Mat& img);

        void readGrayScaleImg(AlignedVec<T>& imgData, Order order, gzFile &file);
        void readRGBImg(AlignedVec<T>& imgData, Order order, gzFile &file);
        cv::Mat toBGR();
};

template <typename T>
class PCA {
    public:

    // Public API
    PCA() = default;
    
    ImgMatrix<double> performPCA(const ImgMatrix<T>& data, uint16_t n_components = 20);
    ImgMatrix<double> inversePCA(const ImgMatrix<double>& data);


    std::vector<std::vector<double>> principalComponents_;

    
    private:
        std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> statistics_;
        std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> fit(const ImgMatrix<T>& data);
        void transform(ImgMatrix<double>& data, const std::vector<std::vector<double>>& mean, const std::vector<std::vector<double>>& std);
        ImgMatrix<double> calcCovMatrix(const ImgMatrix<double>& data);
        void eigenDecomposition(ImgMatrix<double>& data, uint16_t n_components);
        ImgMatrix<double> project(const ImgMatrix<double>& A); 
        ImgMatrix<double> invProject(const ImgMatrix<double>& A);
};



template <typename T>
ImgMatrix<T>::ImgMatrix(std::string_view filename, uint16_t rows, uint16_t columns, uint8_t channels, Order order)
    : imgPath_{filename}, shape_(rows, columns, channels) {
        auto imgData = readGzipFile(order);
        if(imgData.has_value())
            imgData_ = std::move(imgData.value());
        else
            throw std::runtime_error("Failed to read image data from file: " + std::string(filename));

    }

template <typename T>
ImgMatrix<T>::ImgMatrix(uint16_t rows, uint16_t columns, uint8_t channels, Order order)
    : shape_(rows, columns, channels), imgData_(AlignedVec<T>(rows * columns * channels, 0)) {}


template <typename T>
std::optional<AlignedVec<T>> ImgMatrix<T>::readGzipFile(Order order) {

    auto [rows, columns, channels] = shape_;
    AlignedVec<T> imgData(rows * columns * channels);
    auto fp = gzopen(imgPath_.c_str(), "rb");

    if (!fp) {
        std::cerr << "Input file not available!\n";
        return std::nullopt; 
    }

    try {
        switch (channels){
            case static_cast<uint8_t>(1): 
            readGrayScaleImg(imgData, order, fp);
            break;
            case static_cast<uint8_t>(3):
            readRGBImg(imgData, order, fp);
            break;
            default: 
            std::cerr << "Unsupported number of channels\n";
            return std::nullopt;
        }
        
        gzclose(fp);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        gzclose(fp);

        return std::nullopt;
    }
    
    return imgData;
}

template <typename T>
void ImgMatrix<T>::readGrayScaleImg(AlignedVec<T>& imgData, Order order, gzFile &file){
    auto [rows, columns, channels] = shape_;
    int dim1 = (order == Order::ROW_MAJOR ) ? rows : columns;
    int dim2 = (order == Order::ROW_MAJOR ) ? columns : rows;


    //depending on the storage order type, dim2 is either columns or rows
    //storing contiguously rowise or columnwise (tensor included)
    for (auto i = 0; i < dim1; i++)
        gzread(file, imgData.data() + i * dim2, dim2 * sizeof(T));
   
    if (order == Order::COLUMN_MAJOR)
        imgData = ImgMatrix<T>::transpose(imgData, shape_);
            
}


template <typename T>
void ImgMatrix<T>::readRGBImg(AlignedVec<T>& imgData, Order order, gzFile& file){
    auto [rows, columns, channels] = shape_;
    int dim1 = (order == Order::ROW_MAJOR ) ? rows : columns;
    int dim2 = (order == Order::ROW_MAJOR ) ? columns : rows;
    std::vector<T> mem_buffer(dim2 * channels);

    
    for (auto i {0}; i < dim1; i++) {
        gzread(file, mem_buffer.data(), dim2 * channels * sizeof(T));
        
        // first rows * cols elements are R, next rows * cols are G and last rows * cols are B
        for (auto j {0}; j < dim2; j++) {
            auto R {mem_buffer[j * channels]}, G {mem_buffer[j * channels + 1]}, \
            B {mem_buffer[j * channels + 2]};
            imgData[i * dim2 + j] = R;
            imgData[i * dim2 + j + dim1 * dim2] = G;
            imgData[i * dim2 + j + 2 * dim1 * dim2] = B;
        }
    }

    if (order == Order::COLUMN_MAJOR)
        imgData = ImgMatrix<T>::transpose(imgData, std::make_tuple(columns, rows, channels));
}


template <typename T>
AlignedVec<T> ImgMatrix<T>::transpose(const AlignedVec<T>& data, 
                                       std::tuple<uint16_t, uint16_t, uint8_t> shape_) {   
    auto [rows, columns, channels] = shape_;    
    AlignedVec<T> transpMat(rows * columns * channels);
    
    for (auto k = 0; k < channels; k++)                      
        for (auto j = 0; j < columns; j++) {
            for (auto i = 0; i < rows; i++) {
               transpMat[j * rows + i + k * rows * columns] = data[i * columns + j + k * rows * columns];
        }
    }

                                                               
    return transpMat;
}


template <typename T>
bool ImgMatrix<T>::saveImg(std::string_view outputFilename, ImageFormat fmt){
    
    try {
        switch (fmt){
            case ImageFormat::BIN:
                saveAsBinary(outputFilename);
                break;
            case ImageFormat::BIN_GZIP:
                saveAsCompressedBinary(outputFilename);
                break;
            case ImageFormat::PNG: 
            case ImageFormat::JPG:
                saveAsImg(outputFilename, fmt);
                break;  
            default:
                std::cerr << "Unsupported image format\n";
                return false;
        }
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return false;
    }   
}

template <typename T>
void ImgMatrix<T>::saveAsBinary(std::string_view filename) {
    std::string binaryFilename(filename);
    if (!filename.ends_with(".bin")) {
        
        binaryFilename += ".bin";
    }

    std::ofstream outFile(binaryFilename, std::ios::binary);
    if(!outFile)
        throw std::runtime_error("Error opening file for writing");
    
    outFile.write(reinterpret_cast<const char*>(imgData_.data()), imgData_.size() * sizeof(T));
    outFile.close();
}

template <typename T>
void ImgMatrix<T>::saveAsCompressedBinary(std::string_view filename) {
    std::string compressedBinaryFilename(filename);
    if (!filename.ends_with(".bin.gz")) {
        
        compressedBinaryFilename += ".bin.gz";
    }

    auto outFile = gzopen(compressedBinaryFilename.c_str(), "wb");
    if(!outFile)
        throw std::runtime_error("Error opening file for writing");
    
    gzwrite(outFile, imgData_.data(), imgData_.size() * sizeof(T));
    gzclose(outFile);
}

template<typename T>
void ImgMatrix<T>::saveAsImg(std::string_view filename, ImageFormat fmt) {
    auto [rows, columns, channels] = shape_;
  

    switch (channels){
        case static_cast<uint8_t>(1):{
            cv::Mat image_Gray {rows, columns, CV_64FC1, imgData_.data()};
            writeImg(filename, fmt, image_Gray);
            break;
        }
        case static_cast<uint8_t>(3): {
            cv::Mat image_BGR {toBGR()};
            writeImg(filename, fmt, image_BGR);
            break;
        }
        default:
            throw std::runtime_error("Unsupported number of channels");
            break;
    }
}


template<typename T>
void ImgMatrix<T>::writeImg(std::string_view filename, ImageFormat fmt, cv::Mat& img){

    std::string imgFilename(filename);
    if (fmt == ImageFormat::PNG) {
        if (!(filename.ends_with(".png"))) {
            imgFilename += ".png";
        }
        if (!cv::imwrite(imgFilename, img))
            throw std::runtime_error("Error writing image to file");
        
    }
    else if (fmt == ImageFormat::JPG) {
        if (!(filename.ends_with(".jpg"))) {
            imgFilename += ".jpg";
        }
        if (!cv::imwrite(imgFilename, img, std::vector<int>({cv::IMWRITE_JPEG_QUALITY, img_quality})))
            throw std::runtime_error("Error writing image to file");   
    }
}

template<typename T>
cv::Mat ImgMatrix<T>::toBGR() {
    
    auto [rows, columns, channels] = shape_;
    cv::Mat img_BGR = cv::Mat::zeros(rows, columns, CV_64FC3);

     
    for(auto i {0}; i < rows; i++){
        for(auto j {0}; j < columns; j++){
            auto R {imgData_[i * columns + j]}, G {imgData_[i * columns + j + columns * rows]},\
            B {imgData_[i * columns + j + 2 * columns * rows]};
            img_BGR.at<cv::Vec3d>(i, j)[0] = B;
            img_BGR.at<cv::Vec3d>(i, j)[1] = G;
            img_BGR.at<cv::Vec3d>(i, j)[2] = R;
            
        }
    }   
    return img_BGR;
}



//indexing example: (0,3,5) -> row {0}, column {3}, slice {5}
template<typename T>
const T& ImgMatrix<T>::operator()(size_t i, size_t j, size_t k) const {
    auto [rows, columns, channels] = shape_;
    
    return imgData_[i * columns + j + k * columns * rows];
}

template<typename T>
T& ImgMatrix<T>::operator()(size_t i, size_t j, size_t k) {
    return const_cast<T&>(std::as_const(*this)(i, j, k));
}

template<typename T>
std::tuple<uint16_t, uint16_t, uint8_t> ImgMatrix<T>::shape() const {
    return shape_;
}

template<typename T>
AlignedVec<T>& ImgMatrix<T>::data() {
    return imgData_;
}

template<typename T>
const AlignedVec<T>& ImgMatrix<T>::data() const {
    return imgData_;
}

/*=============================*/
//
//      PCA Implementation
//
/*=============================*/

template <typename T>
ImgMatrix<double> PCA<T>::performPCA(const ImgMatrix<T>& data, uint16_t n_components) {

    if (n_components > std::get<1>(data.shape())) {
        throw std::invalid_argument("Number of components cannot exceed the number of features.");
    }

    
    std::array<int64_t, 5> times{};
    using hclock = std::chrono::high_resolution_clock;  

    std::vector<std::vector<double>> mean, std;
    auto start = hclock::now();
    auto stats_ = fit(data); 
    times[0] = std::chrono::duration_cast<std::chrono::milliseconds>(hclock::now() - start).count();
    
    
    auto [rows, columns, channels] = data.shape();
    ImgMatrix<double> normalizedData(rows, columns, channels);
    std::transform(data.data().begin(), data.data().end(), normalizedData.data().begin(), [](T val) {
        return static_cast<double>(val);
    });
    start = hclock::now();
    transform(normalizedData, std::get<0>(stats_), std::get<1>(stats_));
    times[1] = std::chrono::duration_cast<std::chrono::milliseconds>(hclock::now() - start).count();

    start = hclock::now();
    auto covMatrix = calcCovMatrix(normalizedData);
    times[2] = std::chrono::duration_cast<std::chrono::seconds>(hclock::now() - start).count();

#ifdef _DEBUG_
    reportResults("covarianceMatrix", covMatrix.data(), covMatrix.shape());
#endif
    start = hclock::now();
    eigenDecomposition(covMatrix, n_components);
    times[3] = std::chrono::duration_cast<std::chrono::seconds>(hclock::now() - start).count();
    
    
    start = hclock::now();
    //Projection of the data to the subspace spanned by principal components
    // nxm @ (n_compxm)^T -> nxn_comp
    auto compressedImg = project(normalizedData);
    times[4] = std::chrono::duration_cast<std::chrono::milliseconds>(hclock::now() - start).count();

    statistics_ = std::move(stats_);

#ifdef _DEBUG_
    print_times(times);
    reportResults("mean", std::get<0>(statistics_), std::make_tuple(1, columns, channels));
    reportResults("std", std::get<1>(statistics_), std::make_tuple(1, columns, channels));
    reportResults("eigenVectors", principalComponents_, std::make_tuple(n_components, columns, channels));
    reportResults("projection", compressedImg.data(), compressedImg.shape());
#endif

    return compressedImg;
}

template <typename T>
ImgMatrix<double> PCA<T>::inversePCA(const ImgMatrix<double>& data) {

    using hclock = std::chrono::high_resolution_clock;
    // rowsXn_comp @ (n_compxcolumns_decomp) -> rowsXcolumns_decomp
    
    auto start = hclock::now();
    auto decompressedImg = invProject(data);
    // std::cout << "Inverse projection time: " << std::chrono::duration_cast<std::chrono::milliseconds>(hclock::now() - start).count() << " ms\n";
    auto [rows, columns_decomp, channels] = decompressedImg.shape();      
    
    //denormalize the data
    auto [mean, std] = statistics_;

    start = hclock::now();
    #pragma omp parallel for collapse(3)
    for(auto k = 0; k < channels; k++){
        for(auto i = 0; i < rows; i++){
            for(auto j = 0; j < columns_decomp; j++){
                decompressedImg(i, j, k) = (decompressedImg(i, j, k) * std[k][j]) + mean[k][j];
            }
        }
    }
    

    // std::cout << "Denormalization time: " << std::chrono::duration_cast<std::chrono::milliseconds>(hclock::now() - start).count() << " ms\n";
    std::transform(decompressedImg.data().begin(), decompressedImg.data().end(), decompressedImg.data().begin(), [](double val) {
        return std::clamp(val, 0.0, 255.0);
    });
    return decompressedImg;
    
}

template <typename T>
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> PCA<T>::fit(const ImgMatrix<T>& data){
    auto [rows, columns, channels] = data.shape();
    
    std::vector<std::vector<double>> mean(channels, std::vector<double>(columns, 0.0));
    std::vector<std::vector<double>> std(channels, std::vector<double>(columns, 0.0));
   
    double sum, sum_squared;
    // double msq, prog_mean, delta;
    
    #pragma omp parallel for collapse(2) private(sum, sum_squared)
    for(auto k = 0; k < channels; k++){
        for(auto i = 0; i < columns; i++){
            // Based on the Youngs and Cramer's updating formula
            sum = data(0, i, k); sum_squared = 0.0;     
            for(auto j = 1; j < rows; j++){
                sum += data(j, i, k);
                double norm_factor = 1. / static_cast<double>((j + 1) * j);
                sum_squared += norm_factor * std::pow((j + 1) * data(j, i, k) - sum, 2);
            }
            mean[k][i] = sum  / rows;
            std[k][i] = std::sqrt(sum_squared / rows );
            
            // Alternative approach based on the Welford's online algorithm
            // prog_mean = 0.0; msq = 0.0;
            // for(auto j = 0; j < rows; j++){
            //     delta = data(j, i, k) - prog_mean;
            //     prog_mean += delta / (j + 1);
            //     msq += delta * (data(j, i, k) - prog_mean);
                
            // }
            // mean[k][i] = prog_mean;
            // std[k][i] = std::sqrt(msq / (rows));
        }  
       
    }   
    
    return std::make_pair(mean, std);
}

template <typename T>
void PCA<T>::transform(ImgMatrix<double>& data, 
                       const std::vector<std::vector<double>>& mean, 
                       const std::vector<std::vector<double>>& std) {
                        
    auto [rows, columns, channels] = data.shape();
    
    #pragma omp parallel for collapse(3)
    for(auto k = 0; k < channels; k++){
        for(auto i = 0; i < rows; i++){
            for(auto j = 0; j < columns; j++){
                data(i, j, k) = (data(i, j, k) - mean[k][j]) / std[k][j];
            }
        }
    }
}

template <typename T>
ImgMatrix<double> PCA<T>::calcCovMatrix(const ImgMatrix<double>& data) {
    auto [rows, columns, channels] = data.shape();
    ImgMatrix<double> covMatrix{columns, columns, channels};

  
    auto data_T = ImgMatrix<double>::transpose(data.data(), data.shape());
    double sum;
    // columns are now rows and vice versa
    // X.T @ X = covMatrix | calculation of only the upper triangular part of the symmetric matrix
    #pragma omp parallel for collapse(3)
    for(auto l = 0; l < channels; l++){
        for(auto i = 0; i < columns; i++){
            for(auto j = i; j < columns; j++)
            {
            #ifdef __SIMD__
                __m256d cumprod = _mm256_set1_pd(0.0);
                // __m256d c = _mm256_set1_pd(0.0);
                for(auto k = 0; k < rows; k = k + 4){
                    
                    // op1, op2 are vectors of input matrix data_T
                    __m256d op1 = _mm256_loadu_pd(&data_T[i*rows + k + l*rows*columns]);
                    __m256d op2 = _mm256_loadu_pd(&data_T[j*rows + k + l*rows*columns]);

                    // __m256d prod = _mm256_mul_pd(op1, op2);
                    // __m256d y = _mm256_sub_pd(prod, c);
                    // __m256d t = _mm256_add_pd(cumprod, y);
                    // c = _mm256_sub_pd(_mm256_sub_pd(t, cumprod), y);
                    // cumprod = t;
                   
                    cumprod = _mm256_fmadd_pd(op1, op2, cumprod);

                __builtin_prefetch(&data_T[i*rows + k + l*rows*columns + 4], 0, 3); 
                __builtin_prefetch(&data_T[j*rows + k + l*rows*columns + 4], 0, 1); 


                }
                // reduce the sum of the 4 elements of the vector
                sum = reduce_sum(cumprod);
                
                // remaining elements calculation
                for(auto k = rows - (rows % 4); k < rows; k++){
                    sum += data_T[i*rows + k + l*rows*columns] * data_T[j*rows + k + l*rows*columns];
                }

                covMatrix(i, j, l) = sum / (rows - 1);
                covMatrix(j, i, l) = covMatrix(i, j, l);
            #else
                sum = 0.0;
                for(auto k = 0; k < rows; k++){
                    sum += data(k, i, l) * data(k, j, l);
                }
                covMatrix(i, j, l) = sum / (rows - 1);
                covMatrix(j, i, l) = covMatrix(i, j, l);
            #endif
            }
        }
    }

    
    return covMatrix;

}

template <typename T>
void PCA<T>::eigenDecomposition(ImgMatrix<double>& data, uint16_t  n_components) {
    //Eigen decomposition of the covariance matrix
    //symetric marix, so we can use dsyev. rows == columns
    auto [rows, columns, channels] = data.shape();

    std::vector<double> EigenValues_1xn(columns, 0.0);
    for (auto ch {0}; ch < channels; ch++){
        //data.data() is a reference to the wrapped vector in the ImgMatrix object
        //LAPACK_COL_MAJOR in dsyev ACTUALLY stores eigenvectors in ROW MAJOR ORDER. The opposite holds for LAPACK_ROW_MAJOR
        auto unwrapped_array_ptr = data.data().data() + ch * rows * columns;
        // rows == columns because the matrix A is symmetric
        LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', rows , unwrapped_array_ptr, rows, EigenValues_1xn.data());
    }
    principalComponents_.resize(channels);

    // data.data() = ImgMatrix<double>::transpose(data.data(), std::make_tuple(n_components, columns, channels));
    //Extract n_components principal components out of covMatrix that now stores the eigenvectors.
    auto unwrapped_vec = data.data();
    auto rev_it = unwrapped_vec.end();
    for (auto i = channels - 1; i >= 0; --i) {
        std::advance(rev_it, - n_components * columns);
        auto slice = (channels - 1 - i) * columns * columns;
        principalComponents_[i].insert(principalComponents_[i].end(), rev_it, unwrapped_vec.end() - slice);
        // symmetric matrix. We advance to the next slice of the tensor (aka next matrix of rgb values of the image)
        auto mtx_size = columns * columns;
        std::advance(rev_it, - (mtx_size - n_components * columns));
        
    }

    //Eigenvectors are stored in a flattened array | Row major order.
}

template <typename T>
ImgMatrix<double> PCA<T>::project(const ImgMatrix<double>& A) {
    auto [rows, columns, channels] = A.shape();
    auto n_components = principalComponents_[0].size() / columns;
    ImgMatrix<double> compressedImg(rows, n_components, channels);

    if (channels == 1) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, rows, n_components, columns, 1, \
                        A.data().data(), columns, principalComponents_[0].data(), columns,  \
                        0, compressedImg.data().data(), n_components);
    //Reminder to anyone out of his right mind: LD{A,B,C} := leading dimension of the matrix A,B or C 
    //meaning the distance-stride between two successive rows (row-major case) or columns (column-major case)
    //of the matrix IN memory. In the case of transpose taking place, the leading dimension of the matrix
    // e.g. B is calculated before the transposition.
        return compressedImg;
    }
    
    double sum;
    #pragma omp parallel for collapse(3) private(sum)
    for (auto ch = 0; ch < channels; ch++){
        for(auto i = 0; i < rows; i++){
            for(auto j = 0ul; j < n_components; j++){
                sum = 0.;
                for(size_t k = 0; k < columns; k++){
                    sum += A(i, k, ch) * principalComponents_[ch][j * columns + k];
                }
                compressedImg(i, j, ch) = sum;
            }
        }
    }

    return compressedImg;
}


template <typename T>
ImgMatrix<double> PCA<T>::invProject(const ImgMatrix<double>& A){
    auto [rows, n_pcomp, channels] = A.shape();
    auto columns_decomp = principalComponents_[0].size() / n_pcomp;
    ImgMatrix<double> decompressedImg(rows, columns_decomp, channels);
    if (channels == 1) {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, columns_decomp, n_pcomp, 1, \
                     A.data().data(), n_pcomp, principalComponents_[0].data(), columns_decomp,  \
                     0, decompressedImg.data().data(), columns_decomp);
        
            return decompressedImg;
    }

    #pragma omp parallel for collapse(4)
    for (size_t ch = 0; ch < channels; ch++){
        for(size_t i = 0; i < rows; i++){
            for(size_t j = 0; j < n_pcomp; j++){
                for(size_t k = 0; k < columns_decomp; k++)
                    decompressedImg(i, k, ch) += A(i, j, ch) * principalComponents_[ch][j * columns_decomp + k]; 
                }
        }
    }

    return decompressedImg;
}
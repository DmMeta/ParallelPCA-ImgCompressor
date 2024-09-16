FROM ubuntu:latest

LABEL maintainer="Sofotasios Argyris & Metaxakis Dimitris"

RUN apt update && apt install -y git nano build-essential gfortran wget zlib1g-dev unzip cmake

COPY data /data 
ENV DEBUG=0 BLAS_VERSION=0.3.28
WORKDIR /tmp/build
RUN wget https://github.com/OpenMathLib/OpenBLAS/releases/download/v${BLAS_VERSION}/OpenBLAS-${BLAS_VERSION}.tar.gz && \
    tar -xzf OpenBLAS-${BLAS_VERSION}.tar.gz && \
    cd OpenBLAS-${BLAS_VERSION} && \
    make USE_OPENMP=1 NUM_THREADS=16 TARGET=ZEN -j $(nproc) && \
    make install USE_OPENMP=1 NUM_THREADS=16

RUN cd ../ && mkdir -p /opt/opencv && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip && \
    unzip opencv.zip && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/opt/opencv ../opencv-4.x && \
    cmake --build . --target install -j $(nproc)



RUN apt autoremove -y && apt clean -y && rm -rf /tmp/build /tmp/opencv.zip /tmp/opencv-4.x

COPY src /opt/ImgCompression/src
WORKDIR /opt/ImgCompression/src
RUN make clean && make debug=$DEBUG
ENTRYPOINT ["./main"]



rm -r ./build
mkdir -p build
cd build
source scl_source enable devtoolset-10
CC=gcc CXX=g++ cmake .. -DCMAKE_BUILD_TYPE=Release
make -j6

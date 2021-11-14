rm -r ./build
mkdir -p build
cd build
cmake ..
cmake --build . -j6

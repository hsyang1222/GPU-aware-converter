rm -r build
cmake -S bind -B build -DCMAKE_INSTALL_PREFIX=.
cmake --build build --parallel --config Release
./build/bin/bind
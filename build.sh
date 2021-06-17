rm -r build
rm bind.so
rm libbind.so
cmake -S bind -B build -DCMAKE_INSTALL_PREFIX=.
cmake --build build --parallel --config Release --target install
cp libbind.so bind.so
conda activate fastmri
python scratch.py
python memory_policy.py
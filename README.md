Install conda env:

```
conda env create -f environment.yml
conda activate cascade
```

Compile:
```
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DBoost_NO_BOOST_CMAKE=ON -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python -DCASCADE_INSTALL_LIBDIR=lib -DCASCADE_BUILD_PYTHON_BINDINGS=yes -DCMAKE_BUILD_TYPE=Debug
make install
```

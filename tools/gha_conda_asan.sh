#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install wget

# Install conda+deps.
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -O mambaforge.sh
export deps_dir=$HOME/local
export PATH="$HOME/mambaforge/bin:$PATH"
bash mambaforge.sh -b -p $HOME/mambaforge
mamba env create -f cascade_devel.yml -q -p $deps_dir
source activate $deps_dir

export CXXFLAGS="$CXXFLAGS -fsanitize=address"

mkdir build
cd build

cmake -G "Ninja" ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DCASCADE_BUILD_TESTS=yes -DBoost_NO_BOOST_CMAKE=ON -DCASCADE_BUILD_PYTHON_BINDINGS=yes

cmake --build . -- -v

ctest -j4 -VV

cmake --build . --target install

cd

ASAN_OPTIONS=detect_leaks=0 LD_PRELOAD=$CONDA_PREFIX/lib/libasan.so python -c "from cascade import test; test.run_test_suite()"

set +e
set +x

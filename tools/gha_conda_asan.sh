#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install wget

# Install conda+deps.
wget https://github.com/conda-forge/miniforge/releases/download/22.9.0-3/Mambaforge-22.9.0-3-Linux-x86_64.sh -O mambaforge.sh
export deps_dir=$HOME/local
export PATH="$HOME/mambaforge/bin:$PATH"
bash mambaforge.sh -b -p $HOME/mambaforge
mamba env create -f cascade_devel.yml -q -p $deps_dir
source activate $deps_dir

export CXXFLAGS="$CXXFLAGS -fsanitize=address"

mkdir build
cd build

cmake -G "Ninja" ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DCASCADE_BUILD_TESTS=yes -DBoost_NO_BOOST_CMAKE=ON

cmake --build .

ctest -j4 -VV

set +e
set +x

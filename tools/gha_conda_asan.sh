#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install wget

# Install conda+deps.
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda config --add channels conda-forge
conda config --set channel_priority strict
conda env create -f cascade_devel.yml -q -p $deps_dir
source activate $deps_dir

export CXXFLAGS="$CXXFLAGS -fsanitize=address"

mkdir build
cd build

cmake -G "Ninja" ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DCASCADE_BUILD_TESTS=yes -DBoost_NO_BOOST_CMAKE=ON

cmake --build . -- -v

ctest -j4 -VV

set +e
set +x

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
# adding the necessary tools for doc building
conda install sphinx myst-nb sphinx-book-theme matplotlib
source activate $deps_dir

# Create the build dir and cd into it.
mkdir build
cd build

# Build cascade.
cmake -G "Ninja" ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCASCADE_BUILD_PYTHON_BINDINGS=yes -DCMAKE_BUILD_TYPE=Release -DCASCADE_BUILD_TESTS=no -DBoost_NO_BOOST_CMAKE=ON
cmake --build . --target install

# Build the documentation.
cd ${GITHUB_WORKSPACE}/doc
make html linkcheck

# Run the doctests.
make doctest;

set +e
set +x
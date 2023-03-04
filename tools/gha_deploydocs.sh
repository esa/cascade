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
# Adding the necessary tools for doc building
source activate $deps_dir
# NOTE: do some explicit version pinning here, as the interdeps
# between several sphinx packages are **very** fragile.
mamba install sphinx='4.5.*' myst-nb='0.17.*' sphinx-book-theme='0.3.*' sphinx-design matplotlib pykep sgp4

# Create the build dir and cd into it.
mkdir build
cd build

# Build cascade.
cmake -G "Ninja" ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCASCADE_BUILD_PYTHON_BINDINGS=yes -DCMAKE_BUILD_TYPE=Release -DCASCADE_BUILD_TESTS=no -DBoost_NO_BOOST_CMAKE=ON
cmake --build . --target install

# Build the documentation.
cd ${GITHUB_WORKSPACE}/doc
make html linkcheck

set +e
set +x

#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e


# Install conda+deps.
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh -O miniforge3.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniforge3/bin:$PATH"
bash miniforge3.sh -b -p $HOME/miniforge3
mamba env create --file=kep3_devel.yml -q -p $deps_dir
source activate $deps_dir

mkdir build
cd build

cmake -G "Ninja" ../ -DCMAKE_INSTALL_PREFIX=$deps_dir -DCMAKE_PREFIX_PATH=$deps_dir -DCMAKE_BUILD_TYPE=Debug -DCASCADE_BUILD_TESTS=yes -DBoost_NO_BOOST_CMAKE=ON -DCASCADE_BUILD_PYTHON_BINDINGS=yes

cmake --build . -- -v

ctest -j4 -VV

cmake --build . --target install

cd

python -c "from cascade import test; test.run_test_suite()"

set +e
set +x

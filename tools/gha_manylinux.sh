#!/usr/bin/env bash

# Echo each command.
set -x

# Exit on error.
set -e

# Report on the environrnt variables used for this build.
echo "CASCADE_PY_BUILD_TYPE: ${CASCADE_PY_BUILD_TYPE}"
echo "GITHUB_REF: ${GITHUB_REF}"
echo "GITHUB_WORKSPACE: ${GITHUB_WORKSPACE}"
# No idea why but this following line seems to be necessary (added: 18/01/2023)
git config --global --add safe.directory ${GITHUB_WORKSPACE}
BRANCH_NAME=`git rev-parse --abbrev-ref HEAD`
echo "BRANCH_NAME: ${BRANCH_NAME}"

# Read for what python wheels have to be built.
if [[ ${CASCADE_PY_BUILD_TYPE} == *38* ]]; then
	PYTHON_DIR="cp38-cp38"
elif [[ ${CASCADE_PY_BUILD_TYPE} == *39* ]]; then
	PYTHON_DIR="cp39-cp39"
elif [[ ${CASCADE_PY_BUILD_TYPE} == *310* ]]; then
	PYTHON_DIR="cp310-cp310"
elif [[ ${CASCADE_PY_BUILD_TYPE} == *311* ]]; then
	PYTHON_DIR="cp311-cp311"
elif [[ ${CASCADE_PY_BUILD_TYPE} == *312* ]]; then
	PYTHON_DIR="cp312-cp312"
else
	echo "Invalid build type: ${CASCADE_PY_BUILD_TYPE}"
	exit 1
fi

# Report the inferred directory whwere python is found.
echo "PYTHON_DIR: ${PYTHON_DIR}"

# Check if this is a release build.
if [[ "${GITHUB_REF}" == "refs/tags/v"* ]]; then
    echo "Tag build detected"
	export CASCADE_PY_RELEASE_BUILD="yes"
else
	echo "Non-tag build detected"
fi

# The heyoka/heyoka.py versions to be used.
export HEYOKA_VERSION="0.21.0"
export HEYOKA_PY_VERSION="0.21.7"

# Python mandatory deps.
/opt/python/${PYTHON_DIR}/bin/pip install heyoka==${HEYOKA_PY_VERSION} numpy

# In the pagmo2/manylinux228_x86_64_with_deps:latest image in dockerhub
# the working directory is /root/install, we will install heyoka there.
cd /root/install

# Install heyoka.
curl -L -o heyoka.tar.gz https://github.com/bluescarni/heyoka/archive/refs/tags/v${HEYOKA_VERSION}.tar.gz
tar xzf heyoka.tar.gz
cd heyoka-${HEYOKA_VERSION}

mkdir build
cd build
cmake -DBoost_NO_BOOST_CMAKE=ON \
    -DHEYOKA_WITH_MPPP=yes \
    -DHEYOKA_WITH_SLEEF=yes \
    -DHEYOKA_ENABLE_IPO=ON \
    -DHEYOKA_FORCE_STATIC_LLVM=yes \
    -DHEYOKA_HIDE_LLVM_SYMBOLS=yes \
    -DCMAKE_BUILD_TYPE=Release ../;
make -j4 install

# Install cascade.py.
cd ${GITHUB_WORKSPACE}
mkdir build
cd build
cmake -DBoost_NO_BOOST_CMAKE=ON \
	-DCMAKE_BUILD_TYPE=Release \
	-DCASCADE_BUILD_PYTHON_BINDINGS=ON \
	-DPython3_EXECUTABLE=/opt/python/${PYTHON_DIR}/bin/python ../;
make -j4 install

# Making the wheel and installing it
cd wheel
# Copy the installed cascade.py files into the current dir.
cp -r `/opt/python/${PYTHON_DIR}/bin/python -c 'import site; print(site.getsitepackages()[0])'`/cascade ./
# Create the wheel and repair it.
# NOTE: this is temporary because some libraries in the docker
# image are installed in lib64 rather than lib and they are
# not picked up properly by the linker.
export LD_LIBRARY_PATH="/usr/local/lib64:/usr/local/lib"
/opt/python/${PYTHON_DIR}/bin/python setup.py bdist_wheel
auditwheel repair dist/esa_cascade* -w ./dist2
# Try to install it and run the tests.
unset LD_LIBRARY_PATH
cd /
/opt/python/${PYTHON_DIR}/bin/pip install ${GITHUB_WORKSPACE}/build/wheel/dist2/esa_cascade*
/opt/python/${PYTHON_DIR}/bin/python -c "import cascade; cascade.test.run_test_suite();"

# Upload to PyPI.
if [[ "${CASCADE_PY_RELEASE_BUILD}" == "yes" ]]; then
	/opt/python/${PYTHON_DIR}/bin/pip install twine
	/opt/python/${PYTHON_DIR}/bin/twine upload -u ci4esa ${GITHUB_WORKSPACE}/build/wheel/dist2/esa_cascade*
fi

set +e
set +x

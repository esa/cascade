cmake_minimum_required(VERSION 3.12.0)

# Set default build type to "Release".
# NOTE: this should be done before the project command since the latter can set
# CMAKE_BUILD_TYPE itself (it does so for nmake).
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING
		"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
	FORCE)
endif()

project(cascade VERSION 0.1 LANGUAGES CXX C)

option(CASCADE_BUILD_BENCHMARKS "Build benchmarks." OFF)

# List of source files.
set(CASCADE_SRC_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sim_basic.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sim_propagate.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sim_bvh.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/logging_impl.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/logging.cpp"
)

# Setup of the cascade shared library.
add_library(cascade SHARED "${CASCADE_SRC_FILES}")
set_property(TARGET cascade PROPERTY VERSION "1.0")
set_property(TARGET cascade PROPERTY SOVERSION 1)
set_target_properties(cascade PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(cascade PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE)

# Ensure that C++20 is employed when both compiling and consuming cascade.
target_compile_features(cascade PUBLIC cxx_std_20)
# Enforce vanilla C++20 when compiling cascade.
set_property(TARGET cascade PROPERTY CXX_EXTENSIONS NO)

target_include_directories(cascade PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

# TBB.
find_package(TBB REQUIRED CONFIG)
target_link_libraries(cascade PRIVATE TBB::tbb)

# Boost.
find_package(Boost 1.60 REQUIRED)
target_link_libraries(cascade PUBLIC Boost::boost)

# xtensor.
find_package(xtensor REQUIRED CONFIG)
target_link_libraries(cascade PUBLIC xtensor)

# fmt.
find_package(fmt CONFIG REQUIRED)
target_link_libraries(cascade PRIVATE fmt::fmt)

# heyoka.
find_package(heyoka CONFIG REQUIRED)
target_link_libraries(cascade PRIVATE heyoka::heyoka)
message(STATUS "heyoka version: ${heyoka_VERSION}")
if(heyoka_VERSION VERSION_LESS 0.17)
    message(FATAL_ERROR "heyoka>=0.17 is required, but heyoka ${heyoka_VERSION} was found instead")
endif()

# spdlog.
find_package(spdlog CONFIG REQUIRED)
target_link_libraries(cascade PRIVATE spdlog::spdlog)

if(CASCADE_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

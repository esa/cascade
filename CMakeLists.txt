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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/yacma")

message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")
message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "cascade version: ${cascade_VERSION}")

# Run the YACMA compiler setup.
include(YACMACompilerLinkerSettings)

# Build options.
option(CASCADE_BUILD_BENCHMARKS "Build benchmarks." OFF)

# Assemble the flags.
set(CASCADE_CXX_FLAGS_DEBUG ${YACMA_CXX_FLAGS} ${YACMA_CXX_FLAGS_DEBUG})
set(CASCADE_CXX_FLAGS_RELEASE ${YACMA_CXX_FLAGS})
if(YACMA_COMPILER_IS_MSVC)
  # On both cl and clang-cl, disable the idiotic minmax macros and enable the bigobj option.
  # Also, enable the WIN32_LEAN_AND_MEAN definition:
  # https://stackoverflow.com/questions/11040133/what-does-defining-win32-lean-and-mean-exclude-exactly
  list(APPEND CASCADE_CXX_FLAGS_DEBUG "-DNOMINMAX" "/bigobj" "-DWIN32_LEAN_AND_MEAN")
  list(APPEND CASCADE_CXX_FLAGS_RELEASE "-DNOMINMAX" "/bigobj" "-DWIN32_LEAN_AND_MEAN")
  if(YACMA_COMPILER_IS_CLANGXX)
    # clang-cl emits various warnings, let's just silence them.
    # NOTE: at one point in the recent past, MSVC added an options similar to GCC's isystem:
    # https://blogs.msdn.microsoft.com/vcblog/2017/12/13/broken-warnings-theory/
    # We probably just need to wait for this to be picked up by CMake/clang-cl. Let's
    # revisit the issue in the future.
    list(APPEND _CASCADE_CLANG_CL_DISABLED_WARNINGS
        "-Wno-unused-variable"
        "-Wno-inconsistent-dllimport"
        "-Wno-unknown-pragmas"
        "-Wno-unused-parameter"
        "-Wno-sign-compare"
        "-Wno-deprecated-declarations"
        "-Wno-deprecated-dynamic-exception-spec"
        "-Wno-old-style-cast"
        "-Wno-sign-conversion"
        "-Wno-non-virtual-dtor"
        "-Wno-deprecated"
        "-Wno-shadow"
        "-Wno-shorten-64-to-32"
        "-Wno-reserved-id-macro"
        "-Wno-undef"
        "-Wno-c++98-compat-pedantic"
        "-Wno-documentation-unknown-command"
        "-Wno-zero-as-null-pointer-constant"
        "-Wno-language-extension-token"
        "-Wno-gnu-anonymous-struct"
        "-Wno-nested-anon-types"
        "-Wno-documentation"
        "-Wno-comma"
        "-Wno-nonportable-system-include-path"
        "-Wno-global-constructors"
        "-Wno-redundant-parens"
        "-Wno-exit-time-destructors"
        "-Wno-missing-noreturn"
        "-Wno-switch-enum"
        "-Wno-covered-switch-default"
        "-Wno-float-equal"
        "-Wno-double-promotion"
        "-Wno-microsoft-enum-value"
        "-Wno-missing-prototypes"
        "-Wno-implicit-fallthrough"
        "-Wno-format-nonliteral"
        "-Wno-cast-qual"
        "-Wno-disabled-macro-expansion"
        "-Wno-unused-private-field"
        "-Wno-unused-template"
        "-Wno-unused-macros"
        "-Wno-extra-semi-stmt"
        "-Wno-c++98-compat")
    list(APPEND CASCADE_CXX_FLAGS_DEBUG ${_CASCADE_CLANG_CL_DISABLED_WARNINGS})
    list(APPEND CASCADE_CXX_FLAGS_RELEASE ${_CASCADE_CLANG_CL_DISABLED_WARNINGS})
    unset(_CASCADE_CLANG_CL_DISABLED_WARNINGS)
  else()
    # Same as above, disable some cl warnings.
    list(APPEND CASCADE_CXX_FLAGS_DEBUG "/wd4459" "/wd4127" "/wd4251")
    list(APPEND CASCADE_CXX_FLAGS_RELEASE "/wd4459" "/wd4127" "/wd4251")
  endif()
  # Enable strict conformance mode, if supported.
  set(CMAKE_REQUIRED_QUIET TRUE)
  check_cxx_compiler_flag("/permissive-" _CASCADE_MSVC_SUPPORTS_STRICT_CONFORMANCE)
  unset(CMAKE_REQUIRED_QUIET)
  if(_CASCADE_MSVC_SUPPORTS_STRICT_CONFORMANCE)
    message(STATUS "The '/permissive-' flag is supported, enabling it.")
    list(APPEND CASCADE_CXX_FLAGS_DEBUG "/permissive-")
    list(APPEND CASCADE_CXX_FLAGS_RELEASE "/permissive-")
  endif()
  unset(_CASCADE_MSVC_SUPPORTS_STRICT_CONFORMANCE)
endif()

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

target_compile_options(cascade PRIVATE
    "$<$<CONFIG:Debug>:${CASCADE_CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CASCADE_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:RelWithDebInfo>:${CASCADE_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:MinSizeRel>:${CASCADE_CXX_FLAGS_RELEASE}>"
)

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

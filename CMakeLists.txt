# NOTE: current minimum heyoka requirement.
cmake_minimum_required(VERSION 3.18.0)

# Set default build type to "Release".
# NOTE: this should be done before the project command since the latter can set
# CMAKE_BUILD_TYPE itself (it does so for nmake).
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING
		"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
	FORCE)
endif()

project(cascade VERSION 0.1.6 LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/yacma")

message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")
message(STATUS "System processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "cascade version: ${cascade_VERSION}")

# Run the YACMA compiler setup.
include(YACMACompilerLinkerSettings)

# Build options.
option(CASCADE_BUILD_TESTS "Build unit tests." OFF)
option(CASCADE_BUILD_BENCHMARKS "Build benchmarks." OFF)
option(CASCADE_BUILD_PYTHON_BINDINGS "Build Python bindings." OFF)

# NOTE: on Unix systems, the correct library installation path
# could be something other than just "lib", such as "lib64",
# "lib32", etc., depending on platform/configuration. Apparently,
# CMake provides this information via the GNUInstallDirs module.
# Let's enable this for now on all Unixes except OSX.
# NOTE: potentially, this could be applicable to Cygwin as well.
#
# https://cmake.org/cmake/help/v3.15/module/GNUInstallDirs.html
# https://cmake.org/pipermail/cmake/2013-July/055375.html
if(UNIX AND NOT APPLE)
    include(GNUInstallDirs)
    set(_CASCADE_INSTALL_LIBDIR_DEFAULT "${CMAKE_INSTALL_LIBDIR}")
else()
    set(_CASCADE_INSTALL_LIBDIR_DEFAULT "lib")
endif()
if(NOT CASCADE_INSTALL_LIBDIR)
    set(CASCADE_INSTALL_LIBDIR "${_CASCADE_INSTALL_LIBDIR_DEFAULT}" CACHE STRING
        "Library installation directory." FORCE)
endif()
mark_as_advanced(CASCADE_INSTALL_LIBDIR)
message(STATUS "Library installation directory: ${CASCADE_INSTALL_LIBDIR}")

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
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sim_broad_phase.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sim_narrow_phase.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sim_jit.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/sim_dynamics.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/logging_impl.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/logging.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/detail/atomic_utils.cpp"
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
# NOTE: need 1.73 for atomic_ref.
find_package(Boost 1.73 REQUIRED)
target_link_libraries(cascade PUBLIC Boost::boost)

# fmt.
find_package(fmt CONFIG REQUIRED)
target_link_libraries(cascade PUBLIC fmt::fmt)

# heyoka.
find_package(heyoka CONFIG REQUIRED)
target_link_libraries(cascade PUBLIC heyoka::heyoka)
message(STATUS "heyoka version: ${heyoka_VERSION}")
if(heyoka_VERSION VERSION_LESS 0.21.0)
    message(FATAL_ERROR "heyoka>=0.21.0 is required, but heyoka ${heyoka_VERSION} was found instead")
endif()

# spdlog.
find_package(spdlog CONFIG REQUIRED)
target_link_libraries(cascade PRIVATE spdlog::spdlog)

# Installation of the header files.
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/cascade" DESTINATION include)
#install(FILES "${CMAKE_CURRENT_BINARY_DIR}/include/cascade/config.hpp" DESTINATION include/cascade)

# Installation of the library.
install(TARGETS cascade
    EXPORT cascade_export
    LIBRARY DESTINATION "${CASCADE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CASCADE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION bin
)

if(CASCADE_BUILD_PYTHON_BINDINGS)
    # Setup the list of Python components we need to locate.
    set(_CASCADE_PY_PYTHON3_COMPONENTS Interpreter Development.Module)

    # NOTE: strictly speaking, we should need only the Interpreter and Development.Module
    # components to build cascade.py. However, in certain setups (e.g., skbuild), it looks
    # like if we specify only Development.Module CMake is unable to correctly locate
    # the Python installation. Hence, as a workaround, if the following line fails
    # and Python3 is *not* found, we try again, this time with the full Development
    # component (rather than only Development.Module). This seems to work around the
    # issue, for now at least.
    find_package(Python3 QUIET COMPONENTS ${_CASCADE_PY_PYTHON3_COMPONENTS})

    if(NOT Python3_FOUND)
        list(POP_BACK _CASCADE_PY_PYTHON3_COMPONENTS)
        list(APPEND _CASCADE_PY_PYTHON3_COMPONENTS Development)
    endif()

    find_package(Python3 QUIET REQUIRED COMPONENTS ${_CASCADE_PY_PYTHON3_COMPONENTS})

    message(STATUS "Python3 interpreter: ${Python3_EXECUTABLE}")
    message(STATUS "Python3 installation directory: ${Python3_SITEARCH}")

    if(${Python3_VERSION_MINOR} LESS 5)
        message(FATAL_ERROR "The minimum supported version of Python is 3.5, but version 3.${Python3_VERSION_MINOR} was found instead")
    endif()

    set(CASCADE_PY_INSTALL_PATH "" CACHE STRING "cascade module installation path")
    mark_as_advanced(CASCADE_PY_INSTALL_PATH)
    unset(_CASCADE_PY_PYTHON3_COMPONENTS)

    # pybind11.
    find_package(pybind11 REQUIRED CONFIG)
    if(${pybind11_VERSION} VERSION_LESS "2.10")
        message(FATAL_ERROR "pybind11 >= 2.10 is required, but version ${pybind11_VERSION} was found instead.")
    endif()
endif()

if(CASCADE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()

if(CASCADE_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

if(CASCADE_BUILD_PYTHON_BINDINGS)
  add_subdirectory(cascade.py)
endif()

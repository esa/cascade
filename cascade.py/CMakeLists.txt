# Configure the version file.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/_version.py.in" "${CMAKE_CURRENT_BINARY_DIR}/_version.py" @ONLY)

# The list of cascade.py's Python files.
set(CASCADE_PY_PYTHON_FILES __init__.py)

# Copy the python files in the current binary dir,
# so that we can import cascade from the build dir.
# NOTE: importing from the build dir will work
# only on single-configuration generators.
foreach(CASCADE_PY_PYTHON_FILE ${CASCADE_PY_PYTHON_FILES})
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${CASCADE_PY_PYTHON_FILE}"
        "${CMAKE_CURRENT_BINARY_DIR}/${CASCADE_PY_PYTHON_FILE}" COPYONLY)
endforeach()

# Core module.
Python3_add_library(core MODULE WITH_SOABI
    core.cpp
    # logging.cpp
)

target_link_libraries(core PRIVATE cascade Boost::boost)
target_link_libraries(core PRIVATE "${pybind11_LIBRARIES}")
target_include_directories(core SYSTEM PRIVATE "${pybind11_INCLUDE_DIR}" "${Python3_INCLUDE_DIRS}")
target_compile_definitions(core PRIVATE "${pybind11_DEFINITIONS}")
target_compile_options(core PRIVATE
    "$<$<CONFIG:Debug>:${CASCADE_CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CASCADE_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:RelWithDebInfo>:${CASCADE_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:MinSizeRel>:${CASCADE_CXX_FLAGS_RELEASE}>"
)
set_target_properties(core PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(core PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE)
target_compile_features(core PRIVATE cxx_std_20)
set_property(TARGET core PROPERTY CXX_EXTENSIONS NO)

# if (HEYOKA_PY_ENABLE_IPO)
#     include(CheckIPOSupported)
#     check_ipo_supported(RESULT _HEYOKA_PY_IPO_RESULT OUTPUT _HEYOKA_PY_IPO_OUTPUT)
#     if (_HEYOKA_PY_IPO_RESULT)
#         message(STATUS "IPO requested and supported, enabling.")
#         set_property(TARGET core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
#     else()
#         message(STATUS "IPO requested, but it is not supported by the compiler:\n${_HEYOKA_PY_IPO_OUTPUT}")
#     endif()
#     unset(_HEYOKA_PY_IPO_RESULT)
#     unset(_HEYOKA_PY_IPO_OUTPUT)
# endif()

# Installation setup.
if(CASCADE_PY_INSTALL_PATH STREQUAL "")
    message(STATUS "cascade.py will be installed in the default location: ${Python3_SITEARCH}")
    set(_CASCADE_PY_INSTALL_DIR "${Python3_SITEARCH}/cascade")
else()
    message(STATUS "cascade.py will be installed in the custom location: ${CASCADE_PY_INSTALL_PATH}")
    set(_CASCADE_PY_INSTALL_DIR "${CASCADE_PY_INSTALL_PATH}/cascade")
endif()

# Install the core module.
install(TARGETS core
    RUNTIME DESTINATION ${_CASCADE_PY_INSTALL_DIR}
    LIBRARY DESTINATION ${_CASCADE_PY_INSTALL_DIR}
)

# Add the Python files.
install(FILES ${CASCADE_PY_PYTHON_FILES} "${CMAKE_CURRENT_BINARY_DIR}/_version.py"
    DESTINATION ${_CASCADE_PY_INSTALL_DIR})

unset(_CASCADE_PY_INSTALL_DIR)

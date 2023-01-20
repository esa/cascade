# Find the xtensor bits.
find_package(xtensor REQUIRED CONFIG)
find_package(xtensor-blas REQUIRED CONFIG)
find_package(Boost 1.69 COMPONENTS program_options REQUIRED)

# Find HDF5.
find_package(HDF5 COMPONENTS C REQUIRED)

function(ADD_CASCADE_BENCHMARK arg1)
  add_executable(${arg1} ${arg1}.cpp)
  target_link_libraries(${arg1} PRIVATE cascade Boost::boost Boost::program_options TBB::tbb xtensor xtensor-blas hdf5::hdf5)
  target_compile_definitions(${arg1} PRIVATE XTENSOR_USE_FLENS_BLAS PRIVATE BOOST_ALLOW_DEPRECATED_HEADERS)
  target_compile_options(${arg1} PRIVATE
    "$<$<CONFIG:Debug>:${CASCADE_CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CASCADE_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:RelWithDebInfo>:${CASCADE_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:MinSizeRel>:${CASCADE_CXX_FLAGS_RELEASE}>"
  )
  # Setup the C++ standard.
  target_compile_features(${arg1} PRIVATE cxx_std_20)
  set_property(TARGET ${arg1} PROPERTY CXX_EXTENSIONS NO)
endfunction()

# Data files needed by some benchmarks.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/data/test_ic_19647.txt" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/data/test_par_19647.txt" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/data/best_fit_density.txt" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/data/test_ic_612813.txt" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/data/test_par_612813.txt" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/data/test_ic_612813.hdf5" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/data/test_par_612813.hdf5" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)

ADD_CASCADE_BENCHMARK(random_leo_pop_kep)
ADD_CASCADE_BENCHMARK(2022_leo_pop)
ADD_CASCADE_BENCHMARK(haumea)

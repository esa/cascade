# Find the xtensor bits.
find_package(xtensor REQUIRED CONFIG)
find_package(xtensor-blas REQUIRED CONFIG)

function(ADD_CASCADE_BENCHMARK arg1)
  add_executable(${arg1} ${arg1}.cpp)
  target_link_libraries(${arg1} PRIVATE cascade Boost::boost TBB::tbb xtensor xtensor-blas)
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

ADD_CASCADE_BENCHMARK(random_leo)
ADD_CASCADE_BENCHMARK(haumea_bench)

# Find the xtensor bits.
find_package(xtensor REQUIRED CONFIG)

add_library(cascade_test STATIC catch_main.cpp)
target_compile_options(cascade_test PRIVATE
  "$<$<CONFIG:Debug>:${CASCADE_CXX_FLAGS_DEBUG}>"
  "$<$<CONFIG:Release>:${CASCADE_CXX_FLAGS_RELEASE}>"
  "$<$<CONFIG:RelWithDebInfo>:${CASCADE_CXX_FLAGS_RELEASE}>"
  "$<$<CONFIG:MinSizeRel>:${CASCADE_CXX_FLAGS_RELEASE}>"
)

# Setup the C++ standard.
target_compile_features(cascade_test PRIVATE cxx_std_20)
set_property(TARGET cascade_test PROPERTY CXX_EXTENSIONS NO)

function(ADD_CASCADE_TESTCASE arg1)
  add_executable(${arg1} ${arg1}.cpp)
  target_link_libraries(${arg1} PRIVATE cascade_test cascade Boost::boost fmt::fmt heyoka::heyoka xtensor)
  target_compile_options(${arg1} PRIVATE
    "$<$<CONFIG:Debug>:${CASCADE_CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${CASCADE_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:RelWithDebInfo>:${CASCADE_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:MinSizeRel>:${CASCADE_CXX_FLAGS_RELEASE}>"
  )
  # Setup the C++ standard.
  target_compile_features(${arg1} PRIVATE cxx_std_20)
  set_property(TARGET ${arg1} PROPERTY CXX_EXTENSIONS NO)
  add_test(${arg1} ${arg1})
endfunction()

# Data files needed by some tests.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/x_lbf.txt" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/y_lbf.txt" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/z_lbf.txt" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/vx_lbf.txt" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/vy_lbf.txt" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/vz_lbf.txt" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/sizes_lbf.txt" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/haumea_00_x.txt" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/haumea_00_y.txt" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/haumea_00_z.txt" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/haumea_00_vx.txt" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/haumea_00_vy.txt" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/haumea_00_vz.txt" "${CMAKE_CURRENT_BINARY_DIR}/" COPYONLY)

ADD_CASCADE_TESTCASE(sim_api)
ADD_CASCADE_TESTCASE(single_collision)
ADD_CASCADE_TESTCASE(same_morton_code)
ADD_CASCADE_TESTCASE(heyoka_comparison)
ADD_CASCADE_TESTCASE(particle_removal)
ADD_CASCADE_TESTCASE(change_ct)
ADD_CASCADE_TESTCASE(largish)
ADD_CASCADE_TESTCASE(last_bit_flip)
ADD_CASCADE_TESTCASE(reentry)
ADD_CASCADE_TESTCASE(domain_exit)
ADD_CASCADE_TESTCASE(interrupt_order)
ADD_CASCADE_TESTCASE(haumea_bug_00)
ADD_CASCADE_TESTCASE(conj_tracking)
ADD_CASCADE_TESTCASE(invalid_state)

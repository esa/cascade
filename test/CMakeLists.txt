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
  target_link_libraries(${arg1} PRIVATE cascade_test cascade Boost::boost fmt::fmt heyoka::heyoka)
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

ADD_CASCADE_TESTCASE(single_collision)
ADD_CASCADE_TESTCASE(same_morton_code)
ADD_CASCADE_TESTCASE(heyoka_comparison)
ADD_CASCADE_TESTCASE(particle_removal)
ADD_CASCADE_TESTCASE(change_ct)

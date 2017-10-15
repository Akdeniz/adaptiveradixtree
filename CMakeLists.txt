cmake_minimum_required(VERSION 3.3)
project(adaptiveradixtree)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR})

set(ART_FILES
  adaptive_radix_tree.hpp
  adaptive_radix_tree_node.hpp
  impl/adaptive_radix_tree.cpp
  impl/adaptive_radix_tree_node.cpp
)

add_executable(artgtest
  ${ART_FILES}
  utils.hpp
  tests/test_adaptive_radix_tree.cpp
)

target_link_libraries(artgtest ${GTEST_BOTH_LIBRARIES})


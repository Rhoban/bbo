cmake_minimum_required(VERSION 3.16.3)
project(rhoban_bbo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++17 -fopenmp")

option(ROSBAN_BBO_BUILD_EXAMPLES OFF)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  )

# Directories containing sources files
set(DIRECTORIES
  src/rhoban_bbo
  )

# Build ALL_SOURCES
set (SOURCES)
set (ALL_SOURCES)
set (PREFIXED_SOURCES)
foreach (DIRECTORY ${DIRECTORIES})
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/${DIRECTORY}")
    include (${DIRECTORY}/sources.cmake)

    foreach (SOURCE ${SOURCES})
        set (PREFIXED_SOURCES ${PREFIXED_SOURCES} ${DIRECTORY}/${SOURCE})
    endforeach (SOURCE)

    set (ALL_SOURCES ${ALL_SOURCES} ${PREFIXED_SOURCES})
endforeach (DIRECTORY)

# Declare the library
add_library(rhoban_bbo ${ALL_SOURCES} )
target_link_libraries(rhoban_bbo PUBLIC 
  rhoban_utils
  rhoban_random
  cmaes
)
target_include_directories(rhoban_bbo PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)


if (ROSBAN_BBO_BUILD_EXAMPLES)
  add_executable(bbo_example examples/bbo_example.cpp)
  target_link_libraries(bbo_example rhoban_bbo)

  add_executable(test_cmaes_limits examples/test_cmaes_limits.cpp)
  target_link_libraries(test_cmaes_limits rhoban_bbo)
endif()

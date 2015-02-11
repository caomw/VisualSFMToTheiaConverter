project(visual_sfm_to_theia_matches)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
INCLUDE(OptimizeCompilerFlags)

find_package(Theia REQUIRED)
INCLUDE_DIRECTORIES(${THEIA_INCLUDE_DIRS})

OptimizeCompilerFlags()

SET(visual_sfm_sources
  FeaturePoints.cpp
  MatchFile.cpp)

ADD_LIBRARY(visual_sfm SHARED ${visual_sfm_sources})

# executables
ADD_EXECUTABLE(export_vsfm_matches export_vsfm_matches.cc)
TARGET_LINK_LIBRARIES(export_vsfm_matches visual_sfm ${THEIA_LIBRARIES})

# Create Executable
add_executable( convert_vsfm_to_theia_reconstruction convert_vsfm_to_theia_reconstruction.cc)
target_link_libraries( convert_vsfm_to_theia_reconstruction ${THEIA_LIBRARIES})
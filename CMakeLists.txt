cmake_minimum_required(VERSION 3.0.2)
project(uvc_camera_driver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-O3 -march=native)

# ROS 依赖
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  image_transport
  cv_bridge
  dynamic_reconfigure
  camera_info_manager
  diagnostic_updater
)

# dynamic_reconfigure
generate_dynamic_reconfigure_options(
  cfg/CameraParams.cfg
)

# libjpeg-turbo
find_package(PkgConfig REQUIRED)
pkg_check_modules(TURBOJPEG REQUIRED libturbojpeg)

# V4L2
pkg_check_modules(V4L2 REQUIRED libv4l2)

# OpenCV (for cv_bridge)
find_package(OpenCV REQUIRED)

# CUDA & NVJPEG (可选)
find_package(CUDA QUIET)
if(CUDA_FOUND)
  find_library(NVJPEG_LIBRARY nvjpeg HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
  if(NVJPEG_LIBRARY)
    message(STATUS "NVJPEG found: ${NVJPEG_LIBRARY}")
    add_definitions(-DHAS_NVJPEG)
  endif()
endif()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs image_transport cv_bridge
                 dynamic_reconfigure camera_info_manager diagnostic_updater
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${TURBOJPEG_INCLUDE_DIRS}
  ${V4L2_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

if(CUDA_FOUND AND NVJPEG_LIBRARY)
  include_directories(${CUDA_INCLUDE_DIRS})
endif()

# 源文件
set(SOURCES
  src/v4l2_camera.cpp
  src/jpeg_decoder_turbo.cpp
  src/camera_info_parser.cpp
  src/camera_thread.cpp
  src/sync_manager.cpp
  src/camera_node.cpp
)

# NVJPEG 源文件 (可选)
if(CUDA_FOUND AND NVJPEG_LIBRARY)
  list(APPEND SOURCES src/jpeg_decoder_nvjpeg.cpp)
endif()

# 库
add_library(${PROJECT_NAME} ${SOURCES})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${TURBOJPEG_LIBRARIES}
  ${V4L2_LIBRARIES}
  ${OpenCV_LIBRARIES}
  pthread
)

if(CUDA_FOUND AND NVJPEG_LIBRARY)
  target_link_libraries(${PROJECT_NAME} ${NVJPEG_LIBRARY} ${CUDA_LIBRARIES})
endif()

# 可执行文件
add_executable(uvc_camera_node src/main.cpp)
add_dependencies(uvc_camera_node ${PROJECT_NAME}_gencfg)
target_link_libraries(uvc_camera_node ${PROJECT_NAME})

# 安装
install(TARGETS ${PROJECT_NAME} uvc_camera_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

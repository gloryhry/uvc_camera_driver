cmake_minimum_required(VERSION 3.0.2)
project(uvc_camera_driver)

# ==============================================================================
# C++ 标准设置
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==============================================================================
# 引入自定义 CMake 模块
# ==============================================================================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# 平台检测
include(PlatformDetect)

# 编译器优化配置
include(CompilerOptimizations)

# 硬件加速检测
include(HardwareAccel)

# ==============================================================================
# ROS 依赖
# ==============================================================================
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

# ==============================================================================
# 第三方库依赖
# ==============================================================================

# libjpeg-turbo (必需)
find_package(PkgConfig REQUIRED)
pkg_check_modules(TURBOJPEG REQUIRED libturbojpeg)

# V4L2 (必需)
pkg_check_modules(V4L2 REQUIRED libv4l2)

# OpenCV (必需)
find_package(OpenCV REQUIRED)

# ==============================================================================
# Catkin 包配置
# ==============================================================================
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs image_transport cv_bridge
                 dynamic_reconfigure camera_info_manager diagnostic_updater
)

# ==============================================================================
# 头文件目录
# ==============================================================================
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${TURBOJPEG_INCLUDE_DIRS}
  ${V4L2_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# 硬件加速头文件目录
if(HWACCEL_INCLUDE_DIRS)
  include_directories(${HWACCEL_INCLUDE_DIRS})
endif()

# ==============================================================================
# 源文件
# ==============================================================================
set(SOURCES
  src/v4l2_camera.cpp
  src/jpeg_decoder_turbo.cpp
  src/camera_info_parser.cpp
  src/camera_thread.cpp
  src/sync_manager.cpp
  src/camera_node.cpp
)

# 添加硬件加速源文件
if(HWACCEL_SOURCES)
  foreach(HWACCEL_SRC ${HWACCEL_SOURCES})
    list(APPEND SOURCES ${HWACCEL_SRC})
  endforeach()
endif()

message(STATUS "[Build] 源文件列表: ${SOURCES}")

# ==============================================================================
# 库构建
# ==============================================================================
add_library(${PROJECT_NAME} ${SOURCES})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencfg)

# 基础库链接
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${TURBOJPEG_LIBRARIES}
  ${V4L2_LIBRARIES}
  ${OpenCV_LIBRARIES}
  pthread
)

# 硬件加速库链接
if(HWACCEL_LIBRARIES)
  target_link_libraries(${PROJECT_NAME} ${HWACCEL_LIBRARIES})
endif()

# ==============================================================================
# 可执行文件
# ==============================================================================
add_executable(uvc_camera_node src/main.cpp)
add_dependencies(uvc_camera_node ${PROJECT_NAME}_gencfg)
target_link_libraries(uvc_camera_node ${PROJECT_NAME})

# ==============================================================================
# 安装规则
# ==============================================================================
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

# ==============================================================================
# 构建信息汇总
# ==============================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "[Build] UVC Camera Driver 构建配置")
message(STATUS "========================================")
message(STATUS "[Build] 目标平台:     ${PLATFORM_NAME}")
message(STATUS "[Build] C++ 标准:     C++${CMAKE_CXX_STANDARD}")
message(STATUS "[Build] NVJPEG:       ${HAS_NVJPEG}")
message(STATUS "[Build] Rockchip MPP: ${HAS_ROCKCHIP_MPP}")
message(STATUS "[Build] OpenCV 版本:  ${OpenCV_VERSION}")
message(STATUS "========================================")
message(STATUS "")

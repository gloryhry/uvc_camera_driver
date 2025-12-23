# ==============================================================================
# HardwareAccel.cmake - 硬件加速检测与配置模块
# ==============================================================================
# 检测并配置各平台的硬件加速支持
# - NVIDIA: CUDA + NVJPEG
# - Rockchip: MPP (Media Process Platform)
# 依赖: PlatformDetect.cmake 必须先包含
# ==============================================================================

# 用户可配置选项
option(ENABLE_NVJPEG "启用 NVIDIA NVJPEG 硬件加速" ON)
option(ENABLE_ROCKCHIP_MPP "启用 Rockchip MPP 硬件加速" ON)

# 初始化变量
set(HAS_NVJPEG OFF)
set(HAS_ROCKCHIP_MPP OFF)
set(HWACCEL_SOURCES "")
set(HWACCEL_INCLUDE_DIRS "")
set(HWACCEL_LIBRARIES "")

# ==============================================================================
# NVIDIA NVJPEG 检测
# ==============================================================================

if(ENABLE_NVJPEG AND (PLATFORM_IS_JETSON OR PLATFORM_IS_X86_64))
    message(STATUS "[HWAccel] 检测 NVIDIA CUDA/NVJPEG...")
    
    find_package(CUDA QUIET)
    
    if(CUDA_FOUND)
        message(STATUS "[HWAccel] CUDA 版本: ${CUDA_VERSION}")
        message(STATUS "[HWAccel] CUDA 路径: ${CUDA_TOOLKIT_ROOT_DIR}")
        
        # 查找 nvjpeg 库
        find_library(NVJPEG_LIBRARY 
            NAMES nvjpeg
            HINTS 
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                ${CUDA_TOOLKIT_ROOT_DIR}/lib
                /usr/local/cuda/lib64
                /usr/lib/aarch64-linux-gnu
        )
        
        # 查找 nvjpeg 头文件
        find_path(NVJPEG_INCLUDE_DIR
            NAMES nvjpeg.h
            HINTS
                ${CUDA_TOOLKIT_ROOT_DIR}/include
                /usr/local/cuda/include
        )
        
        if(NVJPEG_LIBRARY AND NVJPEG_INCLUDE_DIR)
            set(HAS_NVJPEG ON)
            message(STATUS "[HWAccel] NVJPEG 已找到: ${NVJPEG_LIBRARY}")
            
            list(APPEND HWACCEL_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} ${NVJPEG_INCLUDE_DIR})
            list(APPEND HWACCEL_LIBRARIES ${NVJPEG_LIBRARY} ${CUDA_LIBRARIES})
            list(APPEND HWACCEL_SOURCES "src/jpeg_decoder_nvjpeg.cpp")
            
            add_definitions(-DHAS_NVJPEG)
        else()
            message(STATUS "[HWAccel] NVJPEG 未找到，跳过 GPU 加速")
        endif()
    else()
        message(STATUS "[HWAccel] CUDA 未找到，跳过 NVJPEG")
    endif()
endif()

# ==============================================================================
# Rockchip MPP 检测
# ==============================================================================

if(ENABLE_ROCKCHIP_MPP AND PLATFORM_IS_RK3588)
    message(STATUS "[HWAccel] 检测 Rockchip MPP...")
    
    # 查找 MPP 库
    find_library(ROCKCHIP_MPP_LIBRARY
        NAMES rockchip_mpp mpp
        HINTS
            /usr/lib
            /usr/lib/aarch64-linux-gnu
            /usr/local/lib
    )
    
    # 查找 MPP 头文件
    find_path(ROCKCHIP_MPP_INCLUDE_DIR
        NAMES rockchip/rk_mpi.h rk_mpi.h
        HINTS
            /usr/include
            /usr/local/include
            /usr/include/rockchip
    )
    
    # 查找 RGA 库 (用于图像格式转换)
    find_library(ROCKCHIP_RGA_LIBRARY
        NAMES rga
        HINTS
            /usr/lib
            /usr/lib/aarch64-linux-gnu
            /usr/local/lib
    )
    
    if(ROCKCHIP_MPP_LIBRARY AND ROCKCHIP_MPP_INCLUDE_DIR)
        set(HAS_ROCKCHIP_MPP ON)
        message(STATUS "[HWAccel] Rockchip MPP 已找到: ${ROCKCHIP_MPP_LIBRARY}")
        
        list(APPEND HWACCEL_INCLUDE_DIRS ${ROCKCHIP_MPP_INCLUDE_DIR})
        list(APPEND HWACCEL_LIBRARIES ${ROCKCHIP_MPP_LIBRARY})
        list(APPEND HWACCEL_SOURCES "src/jpeg_decoder_mpp.cpp")
        
        # 添加 RGA 库 (如果找到)
        if(ROCKCHIP_RGA_LIBRARY)
            message(STATUS "[HWAccel] Rockchip RGA 已找到: ${ROCKCHIP_RGA_LIBRARY}")
            list(APPEND HWACCEL_LIBRARIES ${ROCKCHIP_RGA_LIBRARY})
            add_definitions(-DHAS_ROCKCHIP_RGA)
        else()
            message(STATUS "[HWAccel] Rockchip RGA 未找到，使用 CPU 格式转换")
        endif()
        
        add_definitions(-DHAS_ROCKCHIP_MPP)
    else()
        message(STATUS "[HWAccel] Rockchip MPP 未找到")
        if(NOT ROCKCHIP_MPP_LIBRARY)
            message(STATUS "[HWAccel]   - 库文件未找到，请安装 librockchip-mpp-dev")
        endif()
        if(NOT ROCKCHIP_MPP_INCLUDE_DIR)
            message(STATUS "[HWAccel]   - 头文件未找到，请安装 librockchip-mpp-dev")
        endif()
    endif()
endif()

# ==============================================================================
# 输出硬件加速检测结果
# ==============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "[HWAccel] 硬件加速配置:")
message(STATUS "[HWAccel]   NVJPEG:       ${HAS_NVJPEG}")
message(STATUS "[HWAccel]   Rockchip MPP: ${HAS_ROCKCHIP_MPP}")
if(HWACCEL_SOURCES)
    message(STATUS "[HWAccel]   额外源文件:  ${HWACCEL_SOURCES}")
endif()
message(STATUS "========================================")
message(STATUS "")

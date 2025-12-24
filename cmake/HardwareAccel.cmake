# ==============================================================================
# HardwareAccel.cmake - 硬件加速检测与配置模块
# ==============================================================================
# 检测并配置各平台的硬件加速支持
# - NVIDIA Jetson: Multimedia API (NVJPG 硬件单元)
# - NVIDIA x86 GPU: CUDA Toolkit nvjpeg
# - Rockchip RK3588: MPP (Media Process Platform)
# 依赖: PlatformDetect.cmake 必须先包含
# ==============================================================================

# 用户可配置选项
option(ENABLE_JETSON_MULTIMEDIA "启用 Jetson Multimedia API 硬件加速" ON)
option(ENABLE_NVJPEG_CUDA "启用 CUDA nvjpeg 硬件加速 (仅 x86)" ON)
option(ENABLE_ROCKCHIP_MPP "启用 Rockchip MPP 硬件加速" ON)

# 初始化变量
set(HAS_JETSON_MULTIMEDIA OFF)
set(HAS_NVJPEG_CUDA OFF)
set(HAS_ROCKCHIP_MPP OFF)
set(HWACCEL_SOURCES "")
set(HWACCEL_INCLUDE_DIRS "")
set(HWACCEL_LIBRARIES "")

# ==============================================================================
# Jetson Multimedia API 检测 (Jetson 平台专用)
# ==============================================================================

if(ENABLE_JETSON_MULTIMEDIA AND PLATFORM_IS_JETSON)
    message(STATUS "[HWAccel] 检测 Jetson Multimedia API...")
    
    # Jetson Multimedia API 路径
    set(JETSON_MULTIMEDIA_API_PATH "/usr/src/jetson_multimedia_api")
    
    # 查找 NvJpegDecoder 头文件
    find_path(JETSON_NVJPEG_INCLUDE_DIR
        NAMES NvJpegDecoder.h
        HINTS
            ${JETSON_MULTIMEDIA_API_PATH}/include
            /usr/include/jetson-multimedia-api
    )
    
    # 查找 libnvjpeg.so (Jetson 版本)
    find_library(JETSON_NVJPEG_LIBRARY
        NAMES nvjpeg
        HINTS
            /usr/lib/aarch64-linux-gnu/tegra
            /usr/lib/aarch64-linux-gnu
    )
    
    # 查找 libnvbufsurface.so
    find_library(NVBUFSURFACE_LIBRARY
        NAMES nvbufsurface
        HINTS
            /usr/lib/aarch64-linux-gnu/tegra
            /usr/lib/aarch64-linux-gnu
    )
    
    # 查找 libnvbufsurftransform.so
    find_library(NVBUFSURFTRANSFORM_LIBRARY
        NAMES nvbufsurftransform
        HINTS
            /usr/lib/aarch64-linux-gnu/tegra
            /usr/lib/aarch64-linux-gnu
    )
    
    # 查找核心的 Jetson 库
    find_library(NVUTILS_LIBRARY
        NAMES nvv4l2 v4l2
        HINTS
            /usr/lib/aarch64-linux-gnu/tegra
            /usr/lib/aarch64-linux-gnu
    )
    
    if(JETSON_NVJPEG_INCLUDE_DIR AND JETSON_NVJPEG_LIBRARY)
        set(HAS_JETSON_MULTIMEDIA ON)
        message(STATUS "[HWAccel] Jetson Multimedia API 已找到")
        message(STATUS "[HWAccel]   头文件: ${JETSON_NVJPEG_INCLUDE_DIR}")
        message(STATUS "[HWAccel]   库文件: ${JETSON_NVJPEG_LIBRARY}")
        
        # ======================================================================
        # JetPack 版本检测 - 用于处理 NvJPEGDecoder Bug
        # JetPack 5.1.2 及以下版本存在解码器缓存 Bug，需要每帧重建解码器
        # JetPack 5.1.3 及以上版本已修复
        # ======================================================================
        set(JETPACK_VERSION_FILE "/etc/nv_tegra_release")
        if(EXISTS ${JETPACK_VERSION_FILE})
            file(READ ${JETPACK_VERSION_FILE} TEGRA_RELEASE_CONTENT)
            # 解析格式: # R35 (release), REVISION: 4.1, ...
            # R35.4.1 = JetPack 5.1.2, R35.5.0 = JetPack 5.1.3
            string(REGEX MATCH "R([0-9]+) \\(release\\), REVISION: ([0-9]+)\\.([0-9]+)" 
                   TEGRA_VERSION_MATCH "${TEGRA_RELEASE_CONTENT}")
            if(TEGRA_VERSION_MATCH)
                set(L4T_MAJOR ${CMAKE_MATCH_1})
                set(L4T_MINOR ${CMAKE_MATCH_2})
                set(L4T_PATCH ${CMAKE_MATCH_3})
                message(STATUS "[HWAccel]   L4T 版本: R${L4T_MAJOR}.${L4T_MINOR}.${L4T_PATCH}")
                
                # 计算版本号用于比较 (R35.4.1 -> 350401)
                math(EXPR L4T_VERSION_NUM "${L4T_MAJOR} * 10000 + ${L4T_MINOR} * 100 + ${L4T_PATCH}")
                
                # R35.5.0 (JetPack 5.1.3) = 350500，此版本及以上已修复 Bug
                if(L4T_VERSION_NUM LESS 350500)
                    message(STATUS "[HWAccel]   检测到 JetPack < 5.1.3，启用 NvJPEGDecoder Bug 修复")
                    add_definitions(-DJETPACK_HAS_NVJPEG_BUG)
                else()
                    message(STATUS "[HWAccel]   检测到 JetPack >= 5.1.3，NvJPEGDecoder Bug 已修复")
                endif()
            else()
                message(STATUS "[HWAccel]   无法解析 L4T 版本，默认启用 NvJPEGDecoder Bug 修复")
                add_definitions(-DJETPACK_HAS_NVJPEG_BUG)
            endif()
        else()
            message(STATUS "[HWAccel]   未找到 nv_tegra_release，默认启用 NvJPEGDecoder Bug 修复")
            add_definitions(-DJETPACK_HAS_NVJPEG_BUG)
        endif()
        
        # ======================================================================
        # libjpeg 头文件配置
        # NvJpegDecoder.cpp 需要 NVIDIA 扩展的 jpeg_decompress_struct
        # (包含 IsVendorbuf, pVendor_buf, fd 等字段)
        # 必须使用 Jetson Multimedia API 自带的 libjpeg-8b 头文件
        # 注意: 必须将 libjpeg-8b 放在包含路径最前面，优先于系统头文件
        # ======================================================================
        
        # Jetson libjpeg-8b 必须在最前面
        list(INSERT HWACCEL_INCLUDE_DIRS 0
            ${JETSON_MULTIMEDIA_API_PATH}/include/libjpeg-8b
            ${JETSON_NVJPEG_INCLUDE_DIR}
        )
        message(STATUS "[HWAccel]   使用 Jetson libjpeg-8b: ${JETSON_MULTIMEDIA_API_PATH}/include/libjpeg-8b")
        
        # 核心 nvjpeg 库
        list(APPEND HWACCEL_LIBRARIES ${JETSON_NVJPEG_LIBRARY})
        
        # NvBufSurface 库
        if(NVBUFSURFACE_LIBRARY)
            list(APPEND HWACCEL_LIBRARIES ${NVBUFSURFACE_LIBRARY})
            message(STATUS "[HWAccel]   nvbufsurface: ${NVBUFSURFACE_LIBRARY}")
        endif()
        
        # NvBufSurfTransform 库
        if(NVBUFSURFTRANSFORM_LIBRARY)
            list(APPEND HWACCEL_LIBRARIES ${NVBUFSURFTRANSFORM_LIBRARY})
            message(STATUS "[HWAccel]   nvbufsurftransform: ${NVBUFSURFTRANSFORM_LIBRARY}")
        endif()
        
        # Jetson Multimedia API 的 NvJpegDecoder 实现需要这些源文件
        set(JETSON_NVJPEG_CLASS_SRC "${JETSON_MULTIMEDIA_API_PATH}/samples/common/classes/NvJpegDecoder.cpp")
        set(JETSON_NVBUFFER_SRC "${JETSON_MULTIMEDIA_API_PATH}/samples/common/classes/NvBuffer.cpp")
        set(JETSON_NVELEMENT_SRC "${JETSON_MULTIMEDIA_API_PATH}/samples/common/classes/NvElement.cpp")
        set(JETSON_NVELEM_PROFILER_SRC "${JETSON_MULTIMEDIA_API_PATH}/samples/common/classes/NvElementProfiler.cpp")
        set(JETSON_NVLOGGING_SRC "${JETSON_MULTIMEDIA_API_PATH}/samples/common/classes/NvLogging.cpp")
        
        if(EXISTS ${JETSON_NVJPEG_CLASS_SRC})
            list(APPEND HWACCEL_SOURCES 
                "src/jpeg_decoder_nvjpeg.cpp"
                ${JETSON_NVJPEG_CLASS_SRC}
                ${JETSON_NVBUFFER_SRC}
                ${JETSON_NVELEMENT_SRC}
                ${JETSON_NVELEM_PROFILER_SRC}
                ${JETSON_NVLOGGING_SRC}
            )
            list(APPEND HWACCEL_INCLUDE_DIRS 
                ${JETSON_MULTIMEDIA_API_PATH}/samples/common/classes
            )
            message(STATUS "[HWAccel]   NvJpegDecoder 源文件: ${JETSON_NVJPEG_CLASS_SRC}")
        else()
            message(WARNING "[HWAccel]   NvJpegDecoder.cpp 未找到: ${JETSON_NVJPEG_CLASS_SRC}")
            list(APPEND HWACCEL_SOURCES "src/jpeg_decoder_nvjpeg.cpp")
        endif()
        
        add_definitions(-DHAS_JETSON_MULTIMEDIA)
    else()
        message(STATUS "[HWAccel] Jetson Multimedia API 未找到")
        if(NOT JETSON_NVJPEG_INCLUDE_DIR)
            message(STATUS "[HWAccel]   - 头文件未找到，请确认 jetson_multimedia_api 已安装")
        endif()
        if(NOT JETSON_NVJPEG_LIBRARY)
            message(STATUS "[HWAccel]   - 库文件未找到")
        endif()
    endif()
endif()

# ==============================================================================
# CUDA Toolkit nvjpeg 检测 (x86 NVIDIA GPU)
# ==============================================================================

if(ENABLE_NVJPEG_CUDA AND PLATFORM_IS_X86_64 AND NOT HAS_JETSON_MULTIMEDIA)
    message(STATUS "[HWAccel] 检测 CUDA Toolkit nvjpeg...")
    
    # 使用 CMake 的 FindCUDAToolkit (CMake 3.17+) 或 FindCUDA
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.17")
        find_package(CUDAToolkit QUIET)
        if(CUDAToolkit_FOUND)
            set(CUDA_FOUND TRUE)
            set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
            set(CUDA_LIBRARIES CUDA::cudart)
        endif()
    else()
        find_package(CUDA QUIET)
    endif()
    
    if(CUDA_FOUND)
        message(STATUS "[HWAccel] CUDA 已找到")
        
        # 查找 nvjpeg 库
        find_library(NVJPEG_CUDA_LIBRARY 
            NAMES nvjpeg
            HINTS 
                ${CUDAToolkit_LIBRARY_DIR}
                ${CUDA_TOOLKIT_ROOT_DIR}/lib64
                ${CUDA_TOOLKIT_ROOT_DIR}/lib
                /usr/local/cuda/lib64
        )
        
        # 查找 nvjpeg 头文件
        find_path(NVJPEG_CUDA_INCLUDE_DIR
            NAMES nvjpeg.h
            HINTS
                ${CUDAToolkit_INCLUDE_DIRS}
                ${CUDA_TOOLKIT_ROOT_DIR}/include
                /usr/local/cuda/include
        )
        
        if(NVJPEG_CUDA_LIBRARY AND NVJPEG_CUDA_INCLUDE_DIR)
            set(HAS_NVJPEG_CUDA ON)
            message(STATUS "[HWAccel] CUDA nvjpeg 已找到: ${NVJPEG_CUDA_LIBRARY}")
            
            list(APPEND HWACCEL_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} ${NVJPEG_CUDA_INCLUDE_DIR})
            list(APPEND HWACCEL_LIBRARIES ${NVJPEG_CUDA_LIBRARY} ${CUDA_LIBRARIES})
            list(APPEND HWACCEL_SOURCES "src/jpeg_decoder_nvjpeg.cpp")
            
            add_definitions(-DHAS_NVJPEG_CUDA)
        else()
            message(STATUS "[HWAccel] nvjpeg 库未找到，跳过 CUDA GPU 加速")
        endif()
    else()
        message(STATUS "[HWAccel] CUDA 未找到，跳过 NVJPEG")
    endif()
endif()

# ==============================================================================
# Rockchip MPP 检测 (RK3588)
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
message(STATUS "[HWAccel]   Jetson Multimedia: ${HAS_JETSON_MULTIMEDIA}")
message(STATUS "[HWAccel]   CUDA nvjpeg:       ${HAS_NVJPEG_CUDA}")
message(STATUS "[HWAccel]   Rockchip MPP:      ${HAS_ROCKCHIP_MPP}")
if(HWACCEL_SOURCES)
    message(STATUS "[HWAccel]   额外源文件:       ${HWACCEL_SOURCES}")
endif()
message(STATUS "========================================")
message(STATUS "")

# 兼容性变量 (供主 CMakeLists.txt 使用)
if(HAS_JETSON_MULTIMEDIA OR HAS_NVJPEG_CUDA)
    set(HAS_NVJPEG ON)
else()
    set(HAS_NVJPEG OFF)
endif()

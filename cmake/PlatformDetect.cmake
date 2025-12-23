# ==============================================================================
# PlatformDetect.cmake - 平台自动检测模块
# ==============================================================================
# 检测目标平台并设置相应的变量
# 支持的平台: RK3588, Jetson, x86_64, ARM64, ARM32
# ==============================================================================

# 用户可覆盖的平台选项
set(FORCE_PLATFORM "" CACHE STRING "强制指定目标平台 (RK3588, JETSON, X86_64, ARM64, ARM32)")

# 初始化平台变量
set(PLATFORM_IS_RK3588 OFF)
set(PLATFORM_IS_JETSON OFF)
set(PLATFORM_IS_X86_64 OFF)
set(PLATFORM_IS_ARM64 OFF)
set(PLATFORM_IS_ARM32 OFF)
set(PLATFORM_NAME "Unknown")

# ==============================================================================
# 平台检测逻辑
# ==============================================================================

if(FORCE_PLATFORM)
    # 用户手动指定平台
    message(STATUS "[Platform] 使用用户指定的平台: ${FORCE_PLATFORM}")
    
    if(FORCE_PLATFORM STREQUAL "RK3588")
        set(PLATFORM_IS_RK3588 ON)
        set(PLATFORM_IS_ARM64 ON)
        set(PLATFORM_NAME "RK3588")
    elseif(FORCE_PLATFORM STREQUAL "JETSON")
        set(PLATFORM_IS_JETSON ON)
        set(PLATFORM_IS_ARM64 ON)
        set(PLATFORM_NAME "Jetson")
    elseif(FORCE_PLATFORM STREQUAL "X86_64")
        set(PLATFORM_IS_X86_64 ON)
        set(PLATFORM_NAME "x86_64")
    elseif(FORCE_PLATFORM STREQUAL "ARM64")
        set(PLATFORM_IS_ARM64 ON)
        set(PLATFORM_NAME "ARM64")
    elseif(FORCE_PLATFORM STREQUAL "ARM32")
        set(PLATFORM_IS_ARM32 ON)
        set(PLATFORM_NAME "ARM32")
    else()
        message(WARNING "[Platform] 未知的平台: ${FORCE_PLATFORM}, 使用自动检测")
        set(FORCE_PLATFORM "")
    endif()
endif()

if(NOT FORCE_PLATFORM)
    # 自动检测平台
    message(STATUS "[Platform] 自动检测目标平台...")
    
    # 检测处理器架构
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$")
        set(PLATFORM_IS_X86_64 ON)
        set(PLATFORM_NAME "x86_64")
        message(STATUS "[Platform] 检测到 x86_64 架构")
        
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(PLATFORM_IS_ARM64 ON)
        message(STATUS "[Platform] 检测到 ARM64 架构")
        
        # 进一步检测具体芯片
        # 检测 RK3588
        if(EXISTS "/sys/devices/platform/rockchip-cpuinfo/serial-number" OR
           EXISTS "/proc/device-tree/compatible")
            # 尝试读取兼容性字符串
            if(EXISTS "/proc/device-tree/compatible")
                file(READ "/proc/device-tree/compatible" DEVICE_COMPATIBLE)
                if(DEVICE_COMPATIBLE MATCHES "rk3588")
                    set(PLATFORM_IS_RK3588 ON)
                    set(PLATFORM_NAME "RK3588")
                    message(STATUS "[Platform] 检测到 Rockchip RK3588 平台")
                endif()
            endif()
        endif()
        
        # 检测 NVIDIA Jetson
        if(EXISTS "/etc/nv_tegra_release" OR EXISTS "/proc/device-tree/nvidia,dtsfilename")
            set(PLATFORM_IS_JETSON ON)
            set(PLATFORM_NAME "Jetson")
            message(STATUS "[Platform] 检测到 NVIDIA Jetson 平台")
        endif()
        
        # 如果没有检测到特定平台，保持通用 ARM64
        if(NOT PLATFORM_IS_RK3588 AND NOT PLATFORM_IS_JETSON)
            set(PLATFORM_NAME "ARM64")
            message(STATUS "[Platform] 使用通用 ARM64 配置")
        endif()
        
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|armv7l|armhf)$")
        set(PLATFORM_IS_ARM32 ON)
        set(PLATFORM_NAME "ARM32")
        message(STATUS "[Platform] 检测到 ARM32 架构")
        
    else()
        message(WARNING "[Platform] 未知架构: ${CMAKE_SYSTEM_PROCESSOR}, 使用默认配置")
        set(PLATFORM_NAME "Unknown")
    endif()
endif()

# ==============================================================================
# 输出平台检测结果
# ==============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "[Platform] 目标平台: ${PLATFORM_NAME}")
message(STATUS "[Platform] RK3588: ${PLATFORM_IS_RK3588}")
message(STATUS "[Platform] Jetson: ${PLATFORM_IS_JETSON}")
message(STATUS "[Platform] x86_64: ${PLATFORM_IS_X86_64}")
message(STATUS "[Platform] ARM64:  ${PLATFORM_IS_ARM64}")
message(STATUS "[Platform] ARM32:  ${PLATFORM_IS_ARM32}")
message(STATUS "========================================")
message(STATUS "")

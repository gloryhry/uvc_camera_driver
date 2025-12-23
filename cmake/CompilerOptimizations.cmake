# ==============================================================================
# CompilerOptimizations.cmake - 编译器优化配置模块
# ==============================================================================
# 根据目标平台设置编译器优化标志
# 依赖: PlatformDetect.cmake 必须先包含
# ==============================================================================

# 用户可配置选项
option(ENABLE_NATIVE_ARCH "启用 -march=native (仅用于本机编译)" OFF)
option(ENABLE_NEON "在 ARM 平台上启用 NEON 指令集" ON)
option(ENABLE_AVX "在 x86 平台上启用 AVX 指令集 (自动检测)" ON)
option(ENABLE_LTO "启用链接时优化 (Link-Time Optimization)" OFF)

# 基础优化标志
set(COMMON_OPTIMIZATION_FLAGS "-O3")

# 警告标志
set(COMMON_WARNING_FLAGS "-Wall -Wextra -Wno-unused-parameter")

# ==============================================================================
# 平台特定优化
# ==============================================================================

if(ENABLE_NATIVE_ARCH)
    # 仅用于本机编译，不适用于交叉编译
    message(STATUS "[Compiler] 启用 -march=native (仅限本机编译)")
    set(PLATFORM_ARCH_FLAGS "-march=native")
    
elseif(PLATFORM_IS_RK3588)
    # RK3588: ARM Cortex-A76 (大核) + Cortex-A55 (小核)
    message(STATUS "[Compiler] 使用 RK3588 优化配置 (Cortex-A76 + A55)")
    set(PLATFORM_ARCH_FLAGS "-mcpu=cortex-a76 -mtune=cortex-a76")
    
    if(ENABLE_NEON)
        list(APPEND PLATFORM_ARCH_FLAGS "-mfpu=neon-fp-armv8")
        add_definitions(-DHAS_NEON)
        message(STATUS "[Compiler] NEON 指令集已启用")
    endif()
    
elseif(PLATFORM_IS_JETSON)
    # Jetson Orin NX: ARM Cortex-A78AE
    message(STATUS "[Compiler] 使用 Jetson Orin NX 优化配置 (Cortex-A78AE)")
    set(PLATFORM_ARCH_FLAGS "-mcpu=cortex-a78ae")
    
    if(ENABLE_NEON)
        list(APPEND PLATFORM_ARCH_FLAGS "-mfpu=neon-fp-armv8")
        add_definitions(-DHAS_NEON)
        message(STATUS "[Compiler] NEON 指令集已启用")
    endif()
    
elseif(PLATFORM_IS_X86_64)
    # x86_64 通用优化
    message(STATUS "[Compiler] 使用 x86_64 优化配置")
    set(PLATFORM_ARCH_FLAGS "-mtune=generic -msse4.2")
    
    if(ENABLE_AVX)
        # 检测 AVX 支持
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mavx" COMPILER_SUPPORTS_AVX)
        check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        
        if(COMPILER_SUPPORTS_AVX2)
            list(APPEND PLATFORM_ARCH_FLAGS "-mavx" "-mavx2")
            add_definitions(-DHAS_AVX2)
            message(STATUS "[Compiler] AVX2 指令集已启用")
        elseif(COMPILER_SUPPORTS_AVX)
            list(APPEND PLATFORM_ARCH_FLAGS "-mavx")
            add_definitions(-DHAS_AVX)
            message(STATUS "[Compiler] AVX 指令集已启用")
        endif()
    endif()
    
elseif(PLATFORM_IS_ARM64)
    # 通用 ARM64 优化
    message(STATUS "[Compiler] 使用通用 ARM64 优化配置")
    set(PLATFORM_ARCH_FLAGS "-mcpu=cortex-a53")
    
    if(ENABLE_NEON)
        list(APPEND PLATFORM_ARCH_FLAGS "-mfpu=neon-fp-armv8")
        add_definitions(-DHAS_NEON)
        message(STATUS "[Compiler] NEON 指令集已启用")
    endif()
    
elseif(PLATFORM_IS_ARM32)
    # ARM32 优化
    message(STATUS "[Compiler] 使用 ARM32 优化配置")
    set(PLATFORM_ARCH_FLAGS "-mcpu=cortex-a7 -mfloat-abi=hard")
    
    if(ENABLE_NEON)
        list(APPEND PLATFORM_ARCH_FLAGS "-mfpu=neon")
        add_definitions(-DHAS_NEON)
        message(STATUS "[Compiler] NEON 指令集已启用")
    endif()
    
else()
    # 未知平台，使用保守配置
    message(WARNING "[Compiler] 未知平台，使用保守编译配置")
    set(PLATFORM_ARCH_FLAGS "")
endif()

# ==============================================================================
# 链接时优化 (LTO)
# ==============================================================================

if(ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)
    
    if(LTO_SUPPORTED)
        message(STATUS "[Compiler] 链接时优化 (LTO) 已启用")
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "[Compiler] 编译器不支持 LTO: ${LTO_ERROR}")
    endif()
endif()

# ==============================================================================
# 应用编译标志
# ==============================================================================

# 将列表转换为字符串
string(REPLACE ";" " " PLATFORM_ARCH_FLAGS_STR "${PLATFORM_ARCH_FLAGS}")

# 组合所有编译标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_OPTIMIZATION_FLAGS} ${PLATFORM_ARCH_FLAGS_STR} ${COMMON_WARNING_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_OPTIMIZATION_FLAGS} ${PLATFORM_ARCH_FLAGS_STR} ${COMMON_WARNING_FLAGS}")

message(STATUS "[Compiler] CXX 编译标志: ${CMAKE_CXX_FLAGS}")

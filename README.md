# UVC Camera Driver

基于 V4L2 的 ROS 多相机同步驱动程序，支持多平台硬件加速。

## 功能特性

- **V4L2 零拷贝**：MMAP 内存映射，最小化缓冲区（2 个）
- **JPEG 硬解**：libjpeg-turbo（支持 NVJPEG / RK3588 MPP 硬件加速）
- **多相机同步**：epoll 非阻塞 I/O + 统一时间戳
- **触发模式**：视频流 / 外触发（通过 `zoom_absolute` 控制）
- **性能优化**：CPU 核心绑定、SCHED_FIFO 实时调度
- **动态配置**：dynamic_reconfigure 实时调整参数
- **多平台支持**：自动检测 RK3588 / Jetson / x86 / ARM 平台

## 支持的平台

| 平台 | 架构 | 编译优化 | 硬件加速 |
|-----|-----|---------|---------|
| **RK3588** | ARM64 | Cortex-A76 + NEON | MPP JPEG 解码 |
| **Jetson Orin NX** | ARM64 | Cortex-A78AE + NEON | Multimedia API (NVJPG) |
| **x86_64 + NVIDIA GPU** | x86-64 | SSE4.2 + AVX/AVX2 | CUDA nvjpeg |
| **ARM64 通用** | ARM64 | Cortex-A53 + NEON | 无 |
| **ARM32** | ARM | Cortex-A7 + NEON | 无 |

## 依赖

### 基础依赖

```bash
sudo apt install ros-noetic-image-transport ros-noetic-cv-bridge \
    ros-noetic-dynamic-reconfigure ros-noetic-camera-info-manager \
    ros-noetic-diagnostic-updater libturbojpeg0-dev libv4l-dev
```

### RK3588 硬件加速（可选）

```bash
sudo apt install librockchip-mpp-dev librockchip-mpp1
```

### Jetson Multimedia API（Jetson 设备）

Jetson 设备自带 Multimedia API，位于 `/usr/src/jetson_multimedia_api`。

### CUDA nvjpeg（x86 NVIDIA GPU）

```bash
# 安装 CUDA Toolkit (Ubuntu)
sudo apt install nvidia-cuda-toolkit
```

## 编译

### 标准编译（自动检测平台）

```bash
cd ~/Code/uvc_driver
catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release
source devel/setup.bash
```

### 高级编译选项

```bash
# 强制指定目标平台（用于交叉编译）
catkin_make --cmake-args -DFORCE_PLATFORM=RK3588

# 启用本机优化（仅限本机编译）
catkin_make --cmake-args -DENABLE_NATIVE_ARCH=ON

# 禁用特定硬件加速
catkin_make --cmake-args -DENABLE_NVJPEG=OFF
catkin_make --cmake-args -DENABLE_ROCKCHIP_MPP=OFF
```

### CMake 配置选项

| 选项 | 默认值 | 说明 |
|-----|-------|------|
| `FORCE_PLATFORM` | 自动检测 | 强制指定平台 (RK3588/JETSON/X86_64/ARM64/ARM32) |
| `ENABLE_NATIVE_ARCH` | OFF | 启用 `-march=native` |
| `ENABLE_NEON` | ON | ARM 平台启用 NEON 指令集 |
| `ENABLE_AVX` | ON | x86 平台启用 AVX 指令集 |
| `ENABLE_LTO` | OFF | 启用链接时优化 |
| `ENABLE_JETSON_MULTIMEDIA` | ON | 启用 Jetson Multimedia API |
| `ENABLE_NVJPEG_CUDA` | ON | 启用 CUDA nvjpeg (x86 GPU) |
| `ENABLE_ROCKCHIP_MPP` | ON | 启用 RK3588 MPP 加速 |

## 使用

### 单相机

```bash
roslaunch uvc_camera_driver single_camera.launch device:=/dev/video4
```

参数：
- `device`：相机设备路径
- `camera_info_url`：相机内参文件路径（oST 格式）
- `width`/`height`/`fps`：分辨率和帧率
- `cpu_core`：CPU 核心绑定（-1 = 不绑定）
- `realtime_priority`：实时优先级（0 = 不设置）

### 多相机同步

1. 编辑配置文件 `config/multi_camera.yaml`
2. 启动：
```bash
roslaunch uvc_camera_driver multi_camera_sync.launch
```

### 动态参数

```bash
rosrun rqt_reconfigure rqt_reconfigure
```

可调参数：
- `trigger_mode`：0=视频流，1=外触发
- `exposure_auto`/`exposure_absolute`：曝光控制
- `white_balance_auto`/`white_balance_temperature`：白平衡
- `brightness`/`contrast`/`saturation`/`gamma`/`sharpness`

## 话题

| 话题 | 类型 | 说明 |
|------|------|------|
| `~image_raw` | sensor_msgs/Image | RGB 图像 |
| `~image_raw/compressed` | sensor_msgs/CompressedImage | MJPEG 压缩图像 |
| `~camera_info` | sensor_msgs/CameraInfo | 相机内参 |

## JPEG 解码器优先级

系统会自动选择最佳的 JPEG 解码器：

1. **Rockchip MPP** (RK3588) - 硬件 VPU 解码
2. **Jetson Multimedia API** (Jetson) - NVJPG 硬件单元
3. **CUDA nvjpeg** (x86 NVIDIA GPU) - GPU 加速
4. **libjpeg-turbo** (回退) - CPU SIMD 优化

## 相机内参格式 (oST v5.0)

```
# oST version 5.0 parameters

[image]
width
640
height
400

[narrow_stereo]
camera matrix
fx 0 cx
0 fy cy
0 0 1

distortion
k1 k2 p1 p2

rectification
1 0 0
0 1 0
0 0 1

projection
fx 0 cx 0
0 fy cy 0
0 0 1 0
```

## 项目结构

```
uvc_camera_driver/
├── cmake/                          # CMake 模块
│   ├── PlatformDetect.cmake        # 平台自动检测
│   ├── CompilerOptimizations.cmake # 编译器优化配置
│   └── HardwareAccel.cmake         # 硬件加速检测
├── include/uvc_camera_driver/
│   ├── jpeg_decoder.h              # 解码器抽象接口
│   ├── jpeg_decoder_turbo.h        # libjpeg-turbo 解码器
│   ├── jpeg_decoder_nvjpeg.h       # NVJPEG 解码器
│   ├── jpeg_decoder_mpp.h          # RK3588 MPP 解码器
│   └── ...
├── src/
│   ├── jpeg_decoder_turbo.cpp      # libjpeg-turbo 实现
│   ├── jpeg_decoder_nvjpeg.cpp     # NVJPEG 实现
│   ├── jpeg_decoder_mpp.cpp        # RK3588 MPP 实现
│   └── ...
├── config/                         # 配置文件
├── launch/                         # Launch 文件
└── CMakeLists.txt                  # 主构建文件
```

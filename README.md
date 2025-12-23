# UVC Camera Driver

基于 V4L2 的 ROS 多相机同步驱动程序。

## 功能特性

- **V4L2 零拷贝**：MMAP 内存映射，最小化缓冲区（2 个）
- **JPEG 硬解**：libjpeg-turbo（支持 NVJPEG 扩展）
- **多相机同步**：epoll 非阻塞 I/O + 统一时间戳
- **触发模式**：视频流 / 外触发（通过 `zoom_absolute` 控制）
- **性能优化**：CPU 核心绑定、SCHED_FIFO 实时调度
- **动态配置**：dynamic_reconfigure 实时调整参数

## 依赖

```bash
sudo apt install ros-noetic-image-transport ros-noetic-cv-bridge \
    ros-noetic-dynamic-reconfigure ros-noetic-camera-info-manager \
    ros-noetic-diagnostic-updater libturbojpeg0-dev libv4l-dev
```

## 编译

```bash
cd ~/Code/uvc_driver
catkin_make
source devel/setup.bash
```

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

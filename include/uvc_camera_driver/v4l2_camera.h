#ifndef UVC_CAMERA_DRIVER_V4L2_CAMERA_H
#define UVC_CAMERA_DRIVER_V4L2_CAMERA_H

#include <string>
#include <vector>
#include <memory>
#include <linux/videodev2.h>
#include <sys/mman.h>

namespace uvc_camera_driver {

/**
 * @brief 帧数据结构
 */
struct FrameData {
    void* data;              // 帧数据指针 (零拷贝)
    size_t size;             // 数据大小
    struct timeval timestamp; // V4L2 时间戳
    uint32_t sequence;       // 帧序号
    int buffer_index;        // 缓冲区索引 (用于释放)
};

/**
 * @brief 内存类型枚举
 */
enum class MemoryType {
    DMABUF,  // DMA 缓冲区 (优先)
    MMAP     // 内存映射 (退化)
};

/**
 * @brief V4L2 缓冲区结构
 */
struct V4L2Buffer {
    void* start;
    size_t length;
    int dma_fd;  // DMABUF 文件描述符
};

/**
 * @brief V4L2 相机封装类
 * 
 * 支持 DMABUF/MMAP 零拷贝、非阻塞 I/O、最小化缓冲区
 */
class V4L2Camera {
public:
    V4L2Camera();
    ~V4L2Camera();

    // 禁用拷贝
    V4L2Camera(const V4L2Camera&) = delete;
    V4L2Camera& operator=(const V4L2Camera&) = delete;

    /**
     * @brief 打开相机设备
     * @param device 设备路径 (如 /dev/video4)
     * @return 成功返回 true
     */
    bool open(const std::string& device);

    /**
     * @brief 关闭相机设备
     */
    void close();

    /**
     * @brief 检查设备是否已打开
     */
    bool isOpen() const { return fd_ >= 0; }

    /**
     * @brief 配置视频格式
     * @param width 图像宽度
     * @param height 图像高度
     * @param pixelformat V4L2 像素格式 (默认 MJPEG)
     * @return 成功返回 true
     */
    bool setFormat(int width, int height, uint32_t pixelformat = V4L2_PIX_FMT_MJPEG);

    /**
     * @brief 配置帧率
     * @param fps 帧率
     * @return 成功返回 true
     */
    bool setFrameRate(double fps);

    /**
     * @brief 初始化缓冲区
     * @param count 缓冲区数量 (默认 2，最小化)
     * @return 成功返回 true
     */
    bool initBuffers(int count = 2);

    /**
     * @brief 开始视频流
     * @return 成功返回 true
     */
    bool startStreaming();

    /**
     * @brief 停止视频流
     * @return 成功返回 true
     */
    bool stopStreaming();

    /**
     * @brief 非阻塞方式获取帧
     * @param frame 输出帧数据
     * @return 成功返回 true，无帧可用返回 false
     */
    bool grabFrameNonBlocking(FrameData& frame);

    /**
     * @brief 释放帧缓冲区
     * @param buffer_index 缓冲区索引
     */
    void releaseFrame(int buffer_index);

    /**
     * @brief 设置 V4L2 控制参数
     * @param id 控制 ID (如 V4L2_CID_BRIGHTNESS)
     * @param value 参数值
     * @return 成功返回 true
     */
    bool setControl(uint32_t id, int32_t value);

    /**
     * @brief 获取 V4L2 控制参数
     * @param id 控制 ID
     * @param value 输出参数值
     * @return 成功返回 true
     */
    bool getControl(uint32_t id, int32_t& value);

    /**
     * @brief 设置触发模式 (通过 zoom_absolute)
     * @param external true=外触发, false=视频流
     * @return 成功返回 true
     */
    bool setTriggerMode(bool external);

    /**
     * @brief 获取文件描述符 (用于 epoll)
     */
    int getFd() const { return fd_; }

    /**
     * @brief 获取当前使用的内存类型
     */
    MemoryType getMemoryType() const { return mem_type_; }

    /**
     * @brief 获取设备名称
     */
    const std::string& getDeviceName() const { return device_name_; }

    /**
     * @brief 获取配置的宽度
     */
    int getWidth() const { return width_; }

    /**
     * @brief 获取配置的高度
     */
    int getHeight() const { return height_; }

private:
    /**
     * @brief 检测设备支持的最佳内存类型
     */
    MemoryType detectMemoryType();

    /**
     * @brief 使用 DMABUF 初始化缓冲区
     */
    bool initBuffersDmabuf(int count);

    /**
     * @brief 使用 MMAP 初始化缓冲区
     */
    bool initBuffersMmap(int count);

    /**
     * @brief 释放所有缓冲区
     */
    void freeBuffers();

    int fd_;                          // 设备文件描述符
    std::string device_name_;         // 设备路径
    int width_;                       // 图像宽度
    int height_;                      // 图像高度
    uint32_t pixelformat_;            // 像素格式
    MemoryType mem_type_;             // 内存类型
    std::vector<V4L2Buffer> buffers_; // 缓冲区列表
    bool streaming_;                  // 是否正在流传输
};

}  // namespace uvc_camera_driver

#endif  // UVC_CAMERA_DRIVER_V4L2_CAMERA_H

#ifndef UVC_CAMERA_DRIVER_JPEG_DECODER_H
#define UVC_CAMERA_DRIVER_JPEG_DECODER_H

#include <cstdint>
#include <cstddef>
#include <string>
#include <memory>

namespace uvc_camera_driver {

/**
 * @brief JPEG 解码器抽象接口
 * 
 * 支持 libjpeg-turbo 和 NVJPEG 实现
 */
class JpegDecoder {
public:
    virtual ~JpegDecoder() = default;

    /**
     * @brief 解码 JPEG 数据到 RGB
     * @param src JPEG 源数据
     * @param src_size 源数据大小
     * @param dst 目标缓冲区 (RGB)
     * @param dst_size 目标缓冲区大小
     * @param width 图像宽度
     * @param height 图像高度
     * @return 成功返回 true
     */
    virtual bool decode(const uint8_t* src, size_t src_size,
                       uint8_t* dst, size_t dst_size,
                       int width, int height) = 0;

    /**
     * @brief 获取解码器名称
     */
    virtual std::string getName() const = 0;

    /**
     * @brief 检查解码器是否可用
     */
    virtual bool isAvailable() const = 0;
};

/**
 * @brief 创建最佳可用的 JPEG 解码器
 * 优先使用 NVJPEG，退化到 libjpeg-turbo
 */
std::unique_ptr<JpegDecoder> createBestJpegDecoder();

}  // namespace uvc_camera_driver

#endif  // UVC_CAMERA_DRIVER_JPEG_DECODER_H

#ifndef UVC_CAMERA_DRIVER_JPEG_DECODER_NVJPEG_H
#define UVC_CAMERA_DRIVER_JPEG_DECODER_NVJPEG_H

#include "uvc_camera_driver/jpeg_decoder.h"

#ifdef HAS_NVJPEG

#include <nvjpeg.h>
#include <cuda_runtime.h>

namespace uvc_camera_driver {

/**
 * @brief NVIDIA NVJPEG 硬件加速 JPEG 解码器
 * 
 * 使用 NVIDIA GPU 进行 JPEG 解码，适用于 Jetson 和 x86 NVIDIA GPU 平台
 */
class JpegDecoderNvjpeg : public JpegDecoder {
public:
    JpegDecoderNvjpeg();
    ~JpegDecoderNvjpeg() override;

    // 禁止拷贝
    JpegDecoderNvjpeg(const JpegDecoderNvjpeg&) = delete;
    JpegDecoderNvjpeg& operator=(const JpegDecoderNvjpeg&) = delete;

    /**
     * @brief 解码 JPEG 数据到 RGB
     */
    bool decode(const uint8_t* src, size_t src_size,
                uint8_t* dst, size_t dst_size,
                int width, int height) override;

    /**
     * @brief 获取解码器名称
     */
    std::string getName() const override { return "NVJPEG (GPU)"; }

    /**
     * @brief 检查解码器是否可用
     */
    bool isAvailable() const override { return initialized_; }

private:
    bool initialize();
    void cleanup();

    bool initialized_;
    nvjpegHandle_t nvjpeg_handle_;
    nvjpegJpegState_t jpeg_state_;
    cudaStream_t stream_;
    
    // GPU 内存缓冲区
    nvjpegImage_t output_image_;
    uint8_t* d_output_buffer_;
    size_t output_buffer_size_;
};

}  // namespace uvc_camera_driver

#endif  // HAS_NVJPEG

#endif  // UVC_CAMERA_DRIVER_JPEG_DECODER_NVJPEG_H

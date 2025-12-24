#ifndef UVC_CAMERA_DRIVER_JPEG_DECODER_NVJPEG_H
#define UVC_CAMERA_DRIVER_JPEG_DECODER_NVJPEG_H

#include "uvc_camera_driver/jpeg_decoder.h"

// =============================================================================
// NVIDIA GPU JPEG 解码器
// =============================================================================
// 支持两种实现:
// 1. Jetson 平台: 使用 Jetson Multimedia API 的 NvJPEGDecoder
// 2. x86 NVIDIA GPU: 使用 CUDA Toolkit 的 nvjpeg 库
// =============================================================================

#if defined(HAS_JETSON_MULTIMEDIA)
// -----------------------------------------------------------------------------
// Jetson Multimedia API 实现 (Jetson Orin NX/AGX 等)
// 使用专用的 NVJPG 硬件单元
// 头文件: /usr/src/jetson_multimedia_api/include/
// -----------------------------------------------------------------------------

#include <NvJpegDecoder.h>

namespace uvc_camera_driver {

class JpegDecoderNvjpeg : public JpegDecoder {
public:
    JpegDecoderNvjpeg();
    ~JpegDecoderNvjpeg() override;

    JpegDecoderNvjpeg(const JpegDecoderNvjpeg&) = delete;
    JpegDecoderNvjpeg& operator=(const JpegDecoderNvjpeg&) = delete;

    bool decode(const uint8_t* src, size_t src_size,
                uint8_t* dst, size_t dst_size,
                int width, int height) override;

    std::string getName() const override { return "Jetson NVJPG (Hardware)"; }
    bool isAvailable() const override { return decoder_ != nullptr; }

private:
    NvJPEGDecoder* decoder_;
    
    // YUV 到 RGB 转换
    void convertYUV420ToRGB(const uint8_t* y_plane, const uint8_t* u_plane, 
                            const uint8_t* v_plane, uint8_t* rgb_data,
                            int width, int height, 
                            int y_stride, int uv_stride);
};

}  // namespace uvc_camera_driver

#elif defined(HAS_NVJPEG_CUDA)
// -----------------------------------------------------------------------------
// CUDA Toolkit nvjpeg 实现 (x86 NVIDIA GPU)
// 注意: 此库不适用于 Jetson 平台
// -----------------------------------------------------------------------------

#include <nvjpeg.h>
#include <cuda_runtime.h>

namespace uvc_camera_driver {

class JpegDecoderNvjpeg : public JpegDecoder {
public:
    JpegDecoderNvjpeg();
    ~JpegDecoderNvjpeg() override;

    JpegDecoderNvjpeg(const JpegDecoderNvjpeg&) = delete;
    JpegDecoderNvjpeg& operator=(const JpegDecoderNvjpeg&) = delete;

    bool decode(const uint8_t* src, size_t src_size,
                uint8_t* dst, size_t dst_size,
                int width, int height) override;

    std::string getName() const override { return "NVJPEG CUDA (GPU)"; }
    bool isAvailable() const override { return initialized_; }

private:
    bool initialize();
    void cleanup();

    bool initialized_;
    nvjpegHandle_t nvjpeg_handle_;
    nvjpegJpegState_t jpeg_state_;
    cudaStream_t stream_;
    
    nvjpegImage_t output_image_;
    uint8_t* d_output_buffer_;
    size_t output_buffer_size_;
};

}  // namespace uvc_camera_driver

#endif  // HAS_JETSON_MULTIMEDIA / HAS_NVJPEG_CUDA

#endif  // UVC_CAMERA_DRIVER_JPEG_DECODER_NVJPEG_H

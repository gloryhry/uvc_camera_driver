#include "uvc_camera_driver/jpeg_decoder_turbo.h"
#include <ros/ros.h>

// 硬件加速解码器头文件 (条件包含)
#ifdef HAS_ROCKCHIP_MPP
#include "uvc_camera_driver/jpeg_decoder_mpp.h"
#endif

#if defined(HAS_JETSON_MULTIMEDIA) || defined(HAS_NVJPEG_CUDA)
#include "uvc_camera_driver/jpeg_decoder_nvjpeg.h"
#endif

namespace uvc_camera_driver {

JpegDecoderTurbo::JpegDecoderTurbo()
    : handle_(nullptr) {
    handle_ = tjInitDecompress();
    if (!handle_) {
        ROS_ERROR("Failed to initialize libjpeg-turbo: %s", tjGetErrorStr());
    }
}

JpegDecoderTurbo::~JpegDecoderTurbo() {
    if (handle_) {
        tjDestroy(handle_);
        handle_ = nullptr;
    }
}

bool JpegDecoderTurbo::decode(const uint8_t* src, size_t src_size,
                              uint8_t* dst, size_t dst_size,
                              int width, int height) {
    if (!handle_) {
        return false;
    }

    // Get JPEG image info
    int jpeg_width, jpeg_height, jpeg_subsamp, jpeg_colorspace;
    if (tjDecompressHeader3(handle_, src, src_size,
                            &jpeg_width, &jpeg_height, 
                            &jpeg_subsamp, &jpeg_colorspace) < 0) {
        ROS_WARN("Failed to parse JPEG header: %s", tjGetErrorStr());
        return false;
    }

    // Check buffer size
    size_t required_size = jpeg_width * jpeg_height * 3;  // RGB
    if (dst_size < required_size) {
        ROS_ERROR("Destination buffer too small: %zu < %zu", dst_size, required_size);
        return false;
    }

    // Decode to RGB
    if (tjDecompress2(handle_, src, src_size, dst,
                      jpeg_width, 0, jpeg_height,
                      TJPF_RGB, TJFLAG_FASTDCT | TJFLAG_FASTUPSAMPLE) < 0) {
        ROS_WARN("JPEG decode failed: %s", tjGetErrorStr());
        return false;
    }

    return true;
}

// Factory function implementation
// 优先级: Rockchip MPP > NVJPEG > libjpeg-turbo
std::unique_ptr<JpegDecoder> createBestJpegDecoder() {
    
#ifdef HAS_ROCKCHIP_MPP
    // 优先使用 RK3588 MPP 硬件解码
    {
        auto decoder = std::make_unique<JpegDecoderMpp>();
        if (decoder->isAvailable()) {
            ROS_INFO("Using JPEG decoder: %s", decoder->getName().c_str());
            return decoder;
        }
        ROS_WARN("Rockchip MPP decoder not available, trying alternatives...");
    }
#endif

#if defined(HAS_JETSON_MULTIMEDIA) || defined(HAS_NVJPEG_CUDA)
    // 其次使用 NVIDIA GPU/NVJPG 硬件加速
    {
        auto decoder = std::make_unique<JpegDecoderNvjpeg>();
        if (decoder->isAvailable()) {
            ROS_INFO("Using JPEG decoder: %s", decoder->getName().c_str());
            return decoder;
        }
        ROS_WARN("NVIDIA decoder not available, trying alternatives...");
    }
#endif

    // 回退到 libjpeg-turbo (CPU)
    auto decoder = std::make_unique<JpegDecoderTurbo>();
    if (decoder->isAvailable()) {
        ROS_INFO("Using JPEG decoder: %s", decoder->getName().c_str());
        return decoder;
    }

    ROS_ERROR("No available JPEG decoder");
    return nullptr;
}

}  // namespace uvc_camera_driver

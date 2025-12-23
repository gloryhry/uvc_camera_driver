#include "uvc_camera_driver/jpeg_decoder_turbo.h"
#include <ros/ros.h>

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
std::unique_ptr<JpegDecoder> createBestJpegDecoder() {
#ifdef HAS_NVJPEG
    // Try NVJPEG
    // TODO: Implement NVJPEG detection
#endif

    // Use libjpeg-turbo
    auto decoder = std::make_unique<JpegDecoderTurbo>();
    if (decoder->isAvailable()) {
        ROS_INFO("Using JPEG decoder: %s", decoder->getName().c_str());
        return decoder;
    }

    ROS_ERROR("No available JPEG decoder");
    return nullptr;
}

}  // namespace uvc_camera_driver

#include "uvc_camera_driver/jpeg_decoder_nvjpeg.h"

#ifdef HAS_NVJPEG

#include <ros/ros.h>
#include <cstring>

namespace uvc_camera_driver {

JpegDecoderNvjpeg::JpegDecoderNvjpeg()
    : initialized_(false)
    , nvjpeg_handle_(nullptr)
    , jpeg_state_(nullptr)
    , stream_(nullptr)
    , d_output_buffer_(nullptr)
    , output_buffer_size_(0) {
    
    memset(&output_image_, 0, sizeof(output_image_));
    initialized_ = initialize();
}

JpegDecoderNvjpeg::~JpegDecoderNvjpeg() {
    cleanup();
}

bool JpegDecoderNvjpeg::initialize() {
    // 检查 CUDA 设备
    int device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err != cudaSuccess || device_count == 0) {
        ROS_WARN("No CUDA devices available: %s", cudaGetErrorString(cuda_err));
        return false;
    }

    // 创建 CUDA 流
    cuda_err = cudaStreamCreate(&stream_);
    if (cuda_err != cudaSuccess) {
        ROS_ERROR("Failed to create CUDA stream: %s", cudaGetErrorString(cuda_err));
        return false;
    }

    // 初始化 NVJPEG
    nvjpegStatus_t status = nvjpegCreateSimple(&nvjpeg_handle_);
    if (status != NVJPEG_STATUS_SUCCESS) {
        ROS_ERROR("Failed to create NVJPEG handle: %d", status);
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
        return false;
    }

    // 创建 JPEG 状态
    status = nvjpegJpegStateCreate(nvjpeg_handle_, &jpeg_state_);
    if (status != NVJPEG_STATUS_SUCCESS) {
        ROS_ERROR("Failed to create NVJPEG state: %d", status);
        nvjpegDestroy(nvjpeg_handle_);
        cudaStreamDestroy(stream_);
        nvjpeg_handle_ = nullptr;
        stream_ = nullptr;
        return false;
    }

    ROS_INFO("NVJPEG decoder initialized successfully");
    return true;
}

void JpegDecoderNvjpeg::cleanup() {
    if (d_output_buffer_) {
        cudaFree(d_output_buffer_);
        d_output_buffer_ = nullptr;
    }

    if (jpeg_state_) {
        nvjpegJpegStateDestroy(jpeg_state_);
        jpeg_state_ = nullptr;
    }

    if (nvjpeg_handle_) {
        nvjpegDestroy(nvjpeg_handle_);
        nvjpeg_handle_ = nullptr;
    }

    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }

    output_buffer_size_ = 0;
    initialized_ = false;
}

bool JpegDecoderNvjpeg::decode(const uint8_t* src, size_t src_size,
                               uint8_t* dst, size_t dst_size,
                               int width, int height) {
    if (!initialized_) {
        return false;
    }

    // 获取 JPEG 图像信息
    int num_components;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];

    nvjpegStatus_t status = nvjpegGetImageInfo(
        nvjpeg_handle_, src, src_size,
        &num_components, &subsampling, widths, heights);

    if (status != NVJPEG_STATUS_SUCCESS) {
        ROS_WARN("Failed to get JPEG image info: %d", status);
        return false;
    }

    int jpeg_width = widths[0];
    int jpeg_height = heights[0];

    // 计算所需缓冲区大小
    size_t required_size = jpeg_width * jpeg_height * 3;  // RGB
    if (dst_size < required_size) {
        ROS_ERROR("Destination buffer too small: %zu < %zu", dst_size, required_size);
        return false;
    }

    // 确保 GPU 缓冲区足够大
    if (output_buffer_size_ < required_size) {
        if (d_output_buffer_) {
            cudaFree(d_output_buffer_);
        }
        
        cudaError_t cuda_err = cudaMalloc(&d_output_buffer_, required_size);
        if (cuda_err != cudaSuccess) {
            ROS_ERROR("Failed to allocate GPU buffer: %s", cudaGetErrorString(cuda_err));
            d_output_buffer_ = nullptr;
            output_buffer_size_ = 0;
            return false;
        }
        output_buffer_size_ = required_size;
    }

    // 设置输出图像结构 (使用交错 RGB 格式)
    // NVJPEG 需要分离的通道，但我们使用 RGBI 格式
    output_image_.channel[0] = d_output_buffer_;
    output_image_.pitch[0] = jpeg_width * 3;

    // 解码
    status = nvjpegDecode(nvjpeg_handle_, jpeg_state_,
                          src, src_size,
                          NVJPEG_OUTPUT_RGBI,
                          &output_image_, stream_);

    if (status != NVJPEG_STATUS_SUCCESS) {
        ROS_WARN("NVJPEG decode failed: %d", status);
        return false;
    }

    // 同步并复制到主机内存
    cudaError_t cuda_err = cudaStreamSynchronize(stream_);
    if (cuda_err != cudaSuccess) {
        ROS_ERROR("CUDA stream sync failed: %s", cudaGetErrorString(cuda_err));
        return false;
    }

    cuda_err = cudaMemcpy(dst, d_output_buffer_, required_size, cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) {
        ROS_ERROR("Failed to copy from GPU: %s", cudaGetErrorString(cuda_err));
        return false;
    }

    return true;
}

}  // namespace uvc_camera_driver

#endif  // HAS_NVJPEG

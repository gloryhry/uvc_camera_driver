#include "uvc_camera_driver/jpeg_decoder_nvjpeg.h"

// =============================================================================
// Jetson Multimedia API 实现
// 使用 NVJPG 硬件单元进行 JPEG 解码
// 参考: /usr/src/jetson_multimedia_api/samples/06_jpeg_decode
// =============================================================================

#if defined(HAS_JETSON_MULTIMEDIA)

#include <ros/ros.h>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <NvBuffer.h>
#include <NvUtils.h>


namespace uvc_camera_driver {

JpegDecoderNvjpeg::JpegDecoderNvjpeg()
    : decoder_(nullptr) {
    
    // 创建 NvJPEGDecoder 实例
    decoder_ = NvJPEGDecoder::createJPEGDecoder("jpeg_decoder");
    
    if (!decoder_) {
        ROS_ERROR("Failed to create Jetson NvJPEGDecoder");
    } else {
        ROS_INFO("Jetson NvJPEGDecoder initialized (NVJPG hardware unit)");
    }
}

JpegDecoderNvjpeg::~JpegDecoderNvjpeg() {
    if (decoder_) {
        delete decoder_;
        decoder_ = nullptr;
    }
}

bool JpegDecoderNvjpeg::decode(const uint8_t* src, size_t src_size,
                               uint8_t* dst, size_t dst_size,
                               int width, int height) {
#if defined(JETPACK_HAS_NVJPEG_BUG)
    // ===========================================================================
    // JetPack 5.1.2 及以下版本 Bug Workaround:
    // NvJPEGDecoder 在连续解码相同分辨率的图像时会返回第一帧的缓存数据
    // 解决方案：每次解码前重新创建解码器实例
    // 参考: https://forums.developer.nvidia.com/t/nvjpegdecoder-returns-same-frame
    // ===========================================================================
    
    // 重新创建解码器实例以避免缓存 Bug
    if (decoder_) {
        delete decoder_;
        decoder_ = nullptr;
    }
    decoder_ = NvJPEGDecoder::createJPEGDecoder("jpeg_decoder");
    
    if (!decoder_) {
        ROS_ERROR("Failed to recreate NvJPEGDecoder");
        return false;
    }
#else
    // JetPack 5.1.3+ 版本已修复 Bug，直接复用解码器实例
    if (!decoder_) {
        return false;
    }
#endif

    // 解码参数
    uint32_t pixfmt = 0;
    uint32_t decoded_width = 0;
    uint32_t decoded_height = 0;
    
    // 输出缓冲区
    NvBuffer* buffer = nullptr;
    
    // 调用 NVJPG 硬件解码
    int ret = decoder_->decodeToBuffer(&buffer,
                                       const_cast<unsigned char*>(src),
                                       src_size,
                                       &pixfmt,
                                       &decoded_width,
                                       &decoded_height);
    
    if (ret < 0 || !buffer) {
        ROS_WARN("Jetson NVJPG decode failed: %d", ret);
        return false;
    }

    // 验证解码尺寸
    if (decoded_width == 0 || decoded_height == 0) {
        ROS_WARN("Invalid decoded dimensions: %ux%u", decoded_width, decoded_height);
        delete buffer;
        return false;
    }

    // 检查输出缓冲区大小
    size_t required_size = decoded_width * decoded_height * 3;
    if (dst_size < required_size) {
        ROS_ERROR("Destination buffer too small: %zu < %zu", dst_size, required_size);
        delete buffer;
        return false;
    }
    
    // 验证 buffer planes 是否有效
    if (buffer->n_planes == 0) {
        ROS_WARN("Buffer has no planes");
        delete buffer;
        return false;
    }
    
    // 检查第一个平面的数据是否有效（防止绿色帧）
    if (buffer->planes[0].bytesused == 0) {
        ROS_WARN("Buffer plane 0 has no data (bytesused=0)");
        delete buffer;
        return false;
    }

    // NvBuffer::planes[].data 应该已经包含可访问的数据
    // decodeToBuffer 返回的是软件缓冲区，已通过重建解码器解决缓存 Bug
    if (buffer->n_planes >= 2) {
        // NV12 或 YUV420M 格式 - 多平面
        const uint8_t* y_plane = buffer->planes[0].data;
        const uint8_t* uv_plane = buffer->planes[1].data;
        int y_stride = buffer->planes[0].fmt.stride;
        int uv_stride = buffer->planes[1].fmt.stride;
        
        if (!y_plane || !uv_plane) {
            ROS_ERROR("Buffer planes data is null");
            delete buffer;
            return false;
        }
        
        // 快速检查 Y 平面数据是否有效（采样检测绿色帧）
        // 如果 Y 数据全为 0，会导致绿色帧输出
        bool has_valid_data = false;
        for (uint32_t i = 0; i < decoded_height && !has_valid_data; i += decoded_height / 4) {
            for (uint32_t j = 0; j < decoded_width && !has_valid_data; j += decoded_width / 4) {
                if (y_plane[i * y_stride + j] != 0) {
                    has_valid_data = true;
                }
            }
        }
        if (!has_valid_data) {
            ROS_WARN("Detected invalid frame (Y plane all zeros), skipping");
            delete buffer;
            return false;
        }
        
        // NV12 到 RGB 转换
        convertNV12ToRGB(y_plane, uv_plane, decoded_width, decoded_height,
                         y_stride, uv_stride, dst);
                         
    } else if (buffer->n_planes == 1) {
        // 单平面格式 - Y 和 UV 连续存储
        const uint8_t* data = buffer->planes[0].data;
        int stride = buffer->planes[0].fmt.stride;
        
        if (!data) {
            ROS_ERROR("Buffer plane data is null");
            delete buffer;
            return false;
        }
        
        // 假设是 NV12 格式，Y 和 UV 连续存储
        const uint8_t* y_plane = data;
        const uint8_t* uv_plane = data + stride * decoded_height;
        
        convertNV12ToRGB(y_plane, uv_plane, decoded_width, decoded_height,
                         stride, stride, dst);
    } else {
        ROS_WARN("Unexpected number of planes: %d", buffer->n_planes);
        delete buffer;
        return false;
    }

    delete buffer;
    return true;
}

void JpegDecoderNvjpeg::convertNV12ToRGB(const uint8_t* y_plane, 
                                          const uint8_t* uv_plane,
                                          int width, int height,
                                          int y_stride, int uv_stride,
                                          uint8_t* rgb_data) {
    // NV12: Y 平面 + UV 交错平面
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int y_idx = j * y_stride + i;
            int uv_idx = (j / 2) * uv_stride + (i / 2) * 2;
            
            int y = y_plane[y_idx];
            int u = uv_plane[uv_idx] - 128;
            int v = uv_plane[uv_idx + 1] - 128;

            // BT.601 YUV to RGB
            int r = y + ((359 * v) >> 8);
            int g = y - ((88 * u + 183 * v) >> 8);
            int b = y + ((454 * u) >> 8);

            r = (r < 0) ? 0 : ((r > 255) ? 255 : r);
            g = (g < 0) ? 0 : ((g > 255) ? 255 : g);
            b = (b < 0) ? 0 : ((b > 255) ? 255 : b);

            int rgb_idx = (j * width + i) * 3;
            rgb_data[rgb_idx]     = static_cast<uint8_t>(r);
            rgb_data[rgb_idx + 1] = static_cast<uint8_t>(g);
            rgb_data[rgb_idx + 2] = static_cast<uint8_t>(b);
        }
    }
}

void JpegDecoderNvjpeg::convertYUV420ToRGB(const uint8_t* y_plane, 
                                            const uint8_t* u_plane, 
                                            const uint8_t* v_plane, 
                                            uint8_t* rgb_data,
                                            int width, int height, 
                                            int y_stride, int uv_stride) {
    // I420: Y + U + V 分离平面
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int y_idx = j * y_stride + i;
            int uv_idx = (j / 2) * uv_stride + (i / 2);
            
            int y = y_plane[y_idx];
            int u = u_plane[uv_idx] - 128;
            int v = v_plane[uv_idx] - 128;

            // BT.601 YUV to RGB
            int r = y + ((359 * v) >> 8);
            int g = y - ((88 * u + 183 * v) >> 8);
            int b = y + ((454 * u) >> 8);

            r = (r < 0) ? 0 : ((r > 255) ? 255 : r);
            g = (g < 0) ? 0 : ((g > 255) ? 255 : g);
            b = (b < 0) ? 0 : ((b > 255) ? 255 : b);

            int rgb_idx = (j * width + i) * 3;
            rgb_data[rgb_idx]     = static_cast<uint8_t>(r);
            rgb_data[rgb_idx + 1] = static_cast<uint8_t>(g);
            rgb_data[rgb_idx + 2] = static_cast<uint8_t>(b);
        }
    }
}

}  // namespace uvc_camera_driver

#elif defined(HAS_NVJPEG_CUDA)
// =============================================================================
// CUDA Toolkit nvjpeg 实现 (仅用于 x86 NVIDIA GPU)
// =============================================================================

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
    int device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err != cudaSuccess || device_count == 0) {
        ROS_WARN("No CUDA devices available: %s", cudaGetErrorString(cuda_err));
        return false;
    }

    cuda_err = cudaStreamCreate(&stream_);
    if (cuda_err != cudaSuccess) {
        ROS_ERROR("Failed to create CUDA stream: %s", cudaGetErrorString(cuda_err));
        return false;
    }

    nvjpegStatus_t status = nvjpegCreateSimple(&nvjpeg_handle_);
    if (status != NVJPEG_STATUS_SUCCESS) {
        ROS_ERROR("Failed to create NVJPEG handle: %d", status);
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
        return false;
    }

    status = nvjpegJpegStateCreate(nvjpeg_handle_, &jpeg_state_);
    if (status != NVJPEG_STATUS_SUCCESS) {
        ROS_ERROR("Failed to create NVJPEG state: %d", status);
        nvjpegDestroy(nvjpeg_handle_);
        cudaStreamDestroy(stream_);
        nvjpeg_handle_ = nullptr;
        stream_ = nullptr;
        return false;
    }

    ROS_INFO("CUDA NVJPEG decoder initialized (x86 GPU)");
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
    size_t required_size = jpeg_width * jpeg_height * 3;
    
    if (dst_size < required_size) {
        ROS_ERROR("Destination buffer too small: %zu < %zu", dst_size, required_size);
        return false;
    }

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

    output_image_.channel[0] = d_output_buffer_;
    output_image_.pitch[0] = jpeg_width * 3;

    status = nvjpegDecode(nvjpeg_handle_, jpeg_state_,
                          src, src_size,
                          NVJPEG_OUTPUT_RGBI,
                          &output_image_, stream_);

    if (status != NVJPEG_STATUS_SUCCESS) {
        ROS_WARN("NVJPEG decode failed: %d", status);
        return false;
    }

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

#endif  // HAS_JETSON_MULTIMEDIA / HAS_NVJPEG_CUDA

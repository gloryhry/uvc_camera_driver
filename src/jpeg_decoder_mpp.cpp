#include "uvc_camera_driver/jpeg_decoder_mpp.h"

#ifdef HAS_ROCKCHIP_MPP

#include <ros/ros.h>
#include <cstring>
#include <chrono>

namespace uvc_camera_driver {

JpegDecoderMpp::JpegDecoderMpp()
    : initialized_(false)
    , mpp_ctx_(nullptr)
    , mpp_api_(nullptr)
    , buffer_group_(nullptr)
    , cached_width_(0)
    , cached_height_(0) {
    
    initialized_ = initialize();
}

JpegDecoderMpp::~JpegDecoderMpp() {
    cleanup();
}

bool JpegDecoderMpp::initialize() {
    MPP_RET ret = MPP_OK;

    // 创建 MPP 上下文
    ret = mpp_create(&mpp_ctx_, &mpp_api_);
    if (ret != MPP_OK) {
        ROS_ERROR("Failed to create MPP context: %d", ret);
        return false;
    }

    // 初始化解码器
    MppCodingType coding_type = MPP_VIDEO_CodingMJPEG;
    ret = mpp_init(mpp_ctx_, MPP_CTX_DEC, coding_type);
    if (ret != MPP_OK) {
        ROS_ERROR("Failed to init MPP decoder: %d", ret);
        mpp_destroy(mpp_ctx_);
        mpp_ctx_ = nullptr;
        mpp_api_ = nullptr;
        return false;
    }

    // 配置解码器参数
    MppParam param = nullptr;
    
    // 设置立即输出模式（低延迟）
    RK_U32 immediate_out = 1;
    ret = mpp_api_->control(mpp_ctx_, MPP_DEC_SET_IMMEDIATE_OUT, &immediate_out);
    if (ret != MPP_OK) {
        ROS_WARN("Failed to set immediate output mode: %d", ret);
    }

    // 创建缓冲区组
    ret = mpp_buffer_group_get_internal(&buffer_group_, MPP_BUFFER_TYPE_DRM);
    if (ret != MPP_OK) {
        // 尝试使用普通内存
        ret = mpp_buffer_group_get_internal(&buffer_group_, MPP_BUFFER_TYPE_NORMAL);
        if (ret != MPP_OK) {
            ROS_WARN("Failed to create buffer group: %d", ret);
            buffer_group_ = nullptr;
        }
    }

    ROS_INFO("Rockchip MPP JPEG decoder initialized successfully");
    return true;
}

void JpegDecoderMpp::cleanup() {
    if (buffer_group_) {
        mpp_buffer_group_put(buffer_group_);
        buffer_group_ = nullptr;
    }

    if (mpp_ctx_) {
        mpp_api_->reset(mpp_ctx_);
        mpp_destroy(mpp_ctx_);
        mpp_ctx_ = nullptr;
        mpp_api_ = nullptr;
    }

    initialized_ = false;
}

bool JpegDecoderMpp::decode(const uint8_t* src, size_t src_size,
                            uint8_t* dst, size_t dst_size,
                            int width, int height) {
    if (!initialized_) {
        return false;
    }

    MPP_RET ret = MPP_OK;
    MppPacket packet = nullptr;
    MppFrame frame = nullptr;

    // 创建输入包
    ret = mpp_packet_init(&packet, const_cast<uint8_t*>(src), src_size);
    if (ret != MPP_OK) {
        ROS_WARN("Failed to init MPP packet: %d", ret);
        return false;
    }

    // 设置 EOS 标志（单帧解码）
    mpp_packet_set_eos(packet);

    // 发送数据到解码器
    ret = mpp_api_->decode_put_packet(mpp_ctx_, packet);
    if (ret != MPP_OK) {
        ROS_WARN("Failed to put packet: %d", ret);
        mpp_packet_deinit(&packet);
        return false;
    }

    // 获取解码帧
    int retry_count = 0;
    const int max_retries = 10;
    
    while (retry_count < max_retries) {
        ret = mpp_api_->decode_get_frame(mpp_ctx_, &frame);
        if (ret == MPP_OK && frame) {
            break;
        }
        
        // 短暂等待
        usleep(1000);  // 1ms
        retry_count++;
    }

    mpp_packet_deinit(&packet);

    if (!frame) {
        ROS_WARN("Failed to get decoded frame after %d retries", retry_count);
        return false;
    }

    // 检查帧是否有错误
    if (mpp_frame_get_errinfo(frame) || mpp_frame_get_discard(frame)) {
        ROS_WARN("Decoded frame has errors");
        mpp_frame_deinit(&frame);
        return false;
    }

    // 获取帧信息
    int frame_width = mpp_frame_get_width(frame);
    int frame_height = mpp_frame_get_height(frame);
    int hor_stride = mpp_frame_get_hor_stride(frame);
    int ver_stride = mpp_frame_get_ver_stride(frame);
    MppFrameFormat fmt = mpp_frame_get_fmt(frame);

    // 检查输出缓冲区大小
    size_t required_size = frame_width * frame_height * 3;
    if (dst_size < required_size) {
        ROS_ERROR("Destination buffer too small: %zu < %zu", dst_size, required_size);
        mpp_frame_deinit(&frame);
        return false;
    }

    // 获取帧数据
    MppBuffer buffer = mpp_frame_get_buffer(frame);
    if (!buffer) {
        ROS_WARN("No buffer in decoded frame");
        mpp_frame_deinit(&frame);
        return false;
    }

    uint8_t* frame_data = static_cast<uint8_t*>(mpp_buffer_get_ptr(buffer));

    // 根据输出格式进行转换
    // MPP JPEG 解码通常输出 NV12 或 YUV420
    switch (fmt & MPP_FRAME_FMT_MASK) {
        case MPP_FMT_YUV420SP:      // NV12
        case MPP_FMT_YUV420SP_VU:   // NV21
            convertNV12ToRGB(frame_data, dst, frame_width, frame_height, hor_stride);
            break;
            
        case MPP_FMT_YUV420P:       // I420
            // TODO: 实现 I420 到 RGB 的转换
            ROS_WARN("YUV420P format not yet supported, falling back to NV12 conversion");
            convertNV12ToRGB(frame_data, dst, frame_width, frame_height, hor_stride);
            break;
            
        case MPP_FMT_RGB888:        // 直接 RGB
            memcpy(dst, frame_data, required_size);
            break;
            
        default:
            ROS_WARN("Unsupported frame format: 0x%x", fmt);
            mpp_frame_deinit(&frame);
            return false;
    }

    mpp_frame_deinit(&frame);
    return true;
}

void JpegDecoderMpp::convertNV12ToRGB(const uint8_t* yuv_data, uint8_t* rgb_data,
                                       int width, int height, int stride) {
    // NV12 格式: Y 平面 + 交错的 UV 平面
    const uint8_t* y_plane = yuv_data;
    const uint8_t* uv_plane = yuv_data + stride * height;

    #ifdef HAS_NEON
    // 使用 NEON 加速的转换 (ARM 平台优化)
    // TODO: 实现 NEON 优化版本
    #endif

    // 标准 CPU 实现
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            int y_idx = j * stride + i;
            int uv_idx = (j / 2) * stride + (i / 2) * 2;
            
            int y = y_plane[y_idx];
            int u = uv_plane[uv_idx] - 128;
            int v = uv_plane[uv_idx + 1] - 128;

            // YUV 到 RGB 转换 (BT.601)
            int r = y + ((359 * v) >> 8);
            int g = y - ((88 * u + 183 * v) >> 8);
            int b = y + ((454 * u) >> 8);

            // 钳制到 [0, 255]
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

#endif  // HAS_ROCKCHIP_MPP

#ifndef UVC_CAMERA_DRIVER_JPEG_DECODER_MPP_H
#define UVC_CAMERA_DRIVER_JPEG_DECODER_MPP_H

#include "uvc_camera_driver/jpeg_decoder.h"

#ifdef HAS_ROCKCHIP_MPP

#include <rockchip/rk_mpi.h>
#include <rockchip/mpp_buffer.h>
#include <rockchip/mpp_frame.h>
#include <rockchip/mpp_packet.h>

namespace uvc_camera_driver {

/**
 * @brief Rockchip MPP 硬件加速 JPEG 解码器
 * 
 * 使用 RK3588 的 VPU 硬件进行 JPEG 解码
 * 支持 YUV420/NV12 到 RGB 的格式转换
 */
class JpegDecoderMpp : public JpegDecoder {
public:
    JpegDecoderMpp();
    ~JpegDecoderMpp() override;

    // 禁止拷贝
    JpegDecoderMpp(const JpegDecoderMpp&) = delete;
    JpegDecoderMpp& operator=(const JpegDecoderMpp&) = delete;

    /**
     * @brief 解码 JPEG 数据到 RGB
     */
    bool decode(const uint8_t* src, size_t src_size,
                uint8_t* dst, size_t dst_size,
                int width, int height) override;

    /**
     * @brief 获取解码器名称
     */
    std::string getName() const override { return "Rockchip MPP (Hardware)"; }

    /**
     * @brief 检查解码器是否可用
     */
    bool isAvailable() const override { return initialized_; }

private:
    bool initialize();
    void cleanup();
    
    /**
     * @brief NV12/YUV420 转 RGB24
     * @param yuv_data YUV 数据
     * @param rgb_data RGB 输出缓冲区
     * @param width 图像宽度
     * @param height 图像高度
     * @param stride YUV 行步长
     */
    void convertNV12ToRGB(const uint8_t* yuv_data, uint8_t* rgb_data,
                          int width, int height, int stride);

    bool initialized_;
    MppCtx mpp_ctx_;
    MppApi* mpp_api_;
    MppBufferGroup buffer_group_;
    
    // 帧信息缓存
    int cached_width_;
    int cached_height_;
};

}  // namespace uvc_camera_driver

#endif  // HAS_ROCKCHIP_MPP

#endif  // UVC_CAMERA_DRIVER_JPEG_DECODER_MPP_H

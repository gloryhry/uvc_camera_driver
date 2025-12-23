#ifndef UVC_CAMERA_DRIVER_JPEG_DECODER_TURBO_H
#define UVC_CAMERA_DRIVER_JPEG_DECODER_TURBO_H

#include "uvc_camera_driver/jpeg_decoder.h"
#include <turbojpeg.h>

namespace uvc_camera_driver {

/**
 * @brief libjpeg-turbo JPEG 解码器实现
 */
class JpegDecoderTurbo : public JpegDecoder {
public:
    JpegDecoderTurbo();
    ~JpegDecoderTurbo() override;

    bool decode(const uint8_t* src, size_t src_size,
               uint8_t* dst, size_t dst_size,
               int width, int height) override;

    std::string getName() const override { return "libjpeg-turbo"; }
    bool isAvailable() const override { return handle_ != nullptr; }

private:
    tjhandle handle_;
};

}  // namespace uvc_camera_driver

#endif  // UVC_CAMERA_DRIVER_JPEG_DECODER_TURBO_H

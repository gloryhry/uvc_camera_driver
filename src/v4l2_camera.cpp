#include "uvc_camera_driver/v4l2_camera.h"

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <cstring>
#include <cerrno>

#include <ros/ros.h>

namespace uvc_camera_driver {

V4L2Camera::V4L2Camera()
    : fd_(-1)
    , width_(640)
    , height_(400)
    , pixelformat_(V4L2_PIX_FMT_MJPEG)
    , mem_type_(MemoryType::MMAP)
    , streaming_(false) {
}

V4L2Camera::~V4L2Camera() {
    close();
}

bool V4L2Camera::open(const std::string& device) {
    // Close previous device
    if (fd_ >= 0) {
        close();
    }

    // Open device in non-blocking mode
    fd_ = ::open(device.c_str(), O_RDWR | O_NONBLOCK);
    if (fd_ < 0) {
        ROS_ERROR("Failed to open device %s: %s", device.c_str(), strerror(errno));
        return false;
    }

    device_name_ = device;

    // Query device capabilities
    struct v4l2_capability cap;
    if (ioctl(fd_, VIDIOC_QUERYCAP, &cap) < 0) {
        ROS_ERROR("Failed to query device capabilities: %s", strerror(errno));
        close();
        return false;
    }

    // Check video capture capability
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        ROS_ERROR("Device does not support video capture");
        close();
        return false;
    }

    // Check streaming capability
    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        ROS_ERROR("Device does not support streaming");
        close();
        return false;
    }

    ROS_INFO("Opened camera device: %s (%s)", device.c_str(), cap.card);

    // Detect best memory type
    mem_type_ = detectMemoryType();
    ROS_INFO("Using memory type: %s", 
             mem_type_ == MemoryType::DMABUF ? "DMABUF" : "MMAP");

    return true;
}

void V4L2Camera::close() {
    if (streaming_) {
        stopStreaming();
    }

    freeBuffers();

    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }

    device_name_.clear();
}

MemoryType V4L2Camera::detectMemoryType() {
    // Try DMABUF
    struct v4l2_requestbuffers req = {};
    req.count = 1;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_DMABUF;

    if (ioctl(fd_, VIDIOC_REQBUFS, &req) == 0 && req.count > 0) {
        // Release test buffer
        req.count = 0;
        ioctl(fd_, VIDIOC_REQBUFS, &req);
        return MemoryType::DMABUF;
    }

    // Fallback to MMAP
    return MemoryType::MMAP;
}

bool V4L2Camera::setFormat(int width, int height, uint32_t pixelformat) {
    struct v4l2_format fmt = {};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = pixelformat;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) {
        ROS_ERROR("Failed to set video format: %s", strerror(errno));
        return false;
    }

    // Save actual format
    width_ = fmt.fmt.pix.width;
    height_ = fmt.fmt.pix.height;
    pixelformat_ = fmt.fmt.pix.pixelformat;

    ROS_INFO("Video format: %dx%d, pixel format: %c%c%c%c",
             width_, height_,
             pixelformat_ & 0xFF,
             (pixelformat_ >> 8) & 0xFF,
             (pixelformat_ >> 16) & 0xFF,
             (pixelformat_ >> 24) & 0xFF);

    return true;
}

bool V4L2Camera::setFrameRate(double fps) {
    struct v4l2_streamparm parm = {};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    // Get current parameters
    if (ioctl(fd_, VIDIOC_G_PARM, &parm) < 0) {
        ROS_WARN("Failed to get stream parameters: %s", strerror(errno));
        return false;
    }

    // Check if frame rate setting is supported
    if (!(parm.parm.capture.capability & V4L2_CAP_TIMEPERFRAME)) {
        ROS_WARN("Device does not support frame rate setting");
        return false;
    }

    // Set frame rate
    parm.parm.capture.timeperframe.numerator = 1;
    parm.parm.capture.timeperframe.denominator = static_cast<uint32_t>(fps);

    if (ioctl(fd_, VIDIOC_S_PARM, &parm) < 0) {
        ROS_WARN("Failed to set frame rate: %s", strerror(errno));
        return false;
    }

    double actual_fps = static_cast<double>(parm.parm.capture.timeperframe.denominator) /
                        parm.parm.capture.timeperframe.numerator;
    ROS_INFO("Frame rate set to: %.2f fps", actual_fps);

    return true;
}

bool V4L2Camera::initBuffers(int count) {
    if (mem_type_ == MemoryType::DMABUF) {
        if (!initBuffersDmabuf(count)) {
            ROS_WARN("DMABUF initialization failed, falling back to MMAP");
            mem_type_ = MemoryType::MMAP;
        }
    }

    if (mem_type_ == MemoryType::MMAP) {
        return initBuffersMmap(count);
    }

    return true;
}

bool V4L2Camera::initBuffersDmabuf(int count) {
    // DMABUF requires external allocator (e.g. DRM/GBM)
    // Simplified implementation, actual project needs DRM integration
    ROS_WARN("DMABUF requires external allocator, skipping");
    return false;
}

bool V4L2Camera::initBuffersMmap(int count) {
    // Request buffers
    struct v4l2_requestbuffers req = {};
    req.count = count;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(fd_, VIDIOC_REQBUFS, &req) < 0) {
        ROS_ERROR("Failed to request buffers: %s", strerror(errno));
        return false;
    }

    if (req.count < 2) {
        ROS_ERROR("Insufficient buffers: %d", req.count);
        return false;
    }

    ROS_INFO("Allocated %d buffers (requested %d)", req.count, count);

    // Map buffers
    buffers_.resize(req.count);
    for (size_t i = 0; i < buffers_.size(); ++i) {
        struct v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (ioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) {
            ROS_ERROR("Failed to query buffer %zu: %s", i, strerror(errno));
            freeBuffers();
            return false;
        }

        buffers_[i].length = buf.length;
        buffers_[i].start = mmap(NULL, buf.length,
                                  PROT_READ | PROT_WRITE,
                                  MAP_SHARED, fd_, buf.m.offset);

        if (buffers_[i].start == MAP_FAILED) {
            ROS_ERROR("Failed to map buffer %zu: %s", i, strerror(errno));
            buffers_[i].start = nullptr;
            freeBuffers();
            return false;
        }

        buffers_[i].dma_fd = -1;
    }

    return true;
}

void V4L2Camera::freeBuffers() {
    for (auto& buf : buffers_) {
        if (buf.start && buf.start != MAP_FAILED) {
            munmap(buf.start, buf.length);
        }
        if (buf.dma_fd >= 0) {
            ::close(buf.dma_fd);
        }
    }
    buffers_.clear();

    // Release V4L2 buffers
    if (fd_ >= 0) {
        struct v4l2_requestbuffers req = {};
        req.count = 0;
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        req.memory = (mem_type_ == MemoryType::DMABUF) ? 
                     V4L2_MEMORY_DMABUF : V4L2_MEMORY_MMAP;
        ioctl(fd_, VIDIOC_REQBUFS, &req);
    }
}

bool V4L2Camera::startStreaming() {
    if (streaming_) {
        return true;
    }

    // Queue all buffers
    for (size_t i = 0; i < buffers_.size(); ++i) {
        struct v4l2_buffer buf = {};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

        if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
            ROS_ERROR("Failed to queue buffer %zu: %s", i, strerror(errno));
            return false;
        }
    }

    // Start streaming
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMON, &type) < 0) {
        ROS_ERROR("Failed to start streaming: %s", strerror(errno));
        return false;
    }

    streaming_ = true;
    ROS_INFO("Camera started capturing");
    return true;
}

bool V4L2Camera::stopStreaming() {
    if (!streaming_) {
        return true;
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd_, VIDIOC_STREAMOFF, &type) < 0) {
        ROS_WARN("Failed to stop streaming: %s", strerror(errno));
    }

    streaming_ = false;
    ROS_INFO("Camera stopped capturing");
    return true;
}

bool V4L2Camera::grabFrameNonBlocking(FrameData& frame) {
    if (!streaming_) {
        return false;
    }

    struct v4l2_buffer buf = {};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    // Non-blocking dequeue
    if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) {
        if (errno == EAGAIN) {
            // No frame available
            return false;
        }
        ROS_ERROR("Failed to dequeue buffer: %s", strerror(errno));
        return false;
    }

    // Fill frame data
    frame.data = buffers_[buf.index].start;
    frame.size = buf.bytesused;
    frame.timestamp = buf.timestamp;
    frame.sequence = buf.sequence;
    frame.buffer_index = buf.index;

    return true;
}

void V4L2Camera::releaseFrame(int buffer_index) {
    if (buffer_index < 0 || static_cast<size_t>(buffer_index) >= buffers_.size()) {
        return;
    }

    struct v4l2_buffer buf = {};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = buffer_index;

    if (ioctl(fd_, VIDIOC_QBUF, &buf) < 0) {
        ROS_WARN("Failed to requeue buffer %d: %s", buffer_index, strerror(errno));
    }
}

bool V4L2Camera::setControl(uint32_t id, int32_t value) {
    struct v4l2_control ctrl = {};
    ctrl.id = id;
    ctrl.value = value;

    if (ioctl(fd_, VIDIOC_S_CTRL, &ctrl) < 0) {
        ROS_WARN("Failed to set control 0x%x: %s", id, strerror(errno));
        return false;
    }

    return true;
}

bool V4L2Camera::getControl(uint32_t id, int32_t& value) {
    struct v4l2_control ctrl = {};
    ctrl.id = id;

    if (ioctl(fd_, VIDIOC_G_CTRL, &ctrl) < 0) {
        ROS_WARN("Failed to get control 0x%x: %s", id, strerror(errno));
        return false;
    }

    value = ctrl.value;
    return true;
}

bool V4L2Camera::setTriggerMode(bool external) {
    // Control trigger mode via zoom_absolute
    // 0 = video stream mode, 1 = external trigger mode
    return setControl(V4L2_CID_ZOOM_ABSOLUTE, external ? 1 : 0);
}

}  // namespace uvc_camera_driver

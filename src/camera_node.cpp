#include "uvc_camera_driver/camera_node.h"

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sys/epoll.h>

namespace uvc_camera_driver {

// ============================================================================
// CameraNode Implementation (Single Camera)
// ============================================================================

CameraNode::CameraNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : nh_(nh)
    , pnh_(pnh)
    , it_(pnh)  // Use private namespace for all topics
    , diagnostics_(nh, pnh)
    , running_(false)
    , frame_count_(0)
    , drop_count_(0) {
}

CameraNode::~CameraNode() {
    stop();
}

bool CameraNode::init() {
    // Load parameters
    loadParameters();

    // Create camera object
    camera_ = std::make_unique<V4L2Camera>();

    // Open camera
    if (!camera_->open(device_)) {
        ROS_ERROR("Failed to open camera: %s", device_.c_str());
        return false;
    }

    // Configure format
    if (!camera_->setFormat(width_, height_, V4L2_PIX_FMT_MJPEG)) {
        ROS_ERROR("Failed to set video format");
        return false;
    }

    // Configure frame rate
    camera_->setFrameRate(fps_);

    // Initialize buffers
    if (!camera_->initBuffers(buffer_count_)) {
        ROS_ERROR("Failed to initialize buffers");
        return false;
    }

    // Create JPEG decoder
    jpeg_decoder_ = createBestJpegDecoder();
    if (!jpeg_decoder_) {
        ROS_ERROR("Failed to create JPEG decoder");
        return false;
    }

    // Allocate RGB buffer
    rgb_buffer_.resize(width_ * height_ * 3);

    // Load camera intrinsics
    if (!camera_info_url_.empty()) {
        if (camera_info_parser_.loadOST(camera_info_url_)) {
            camera_info_msg_ = camera_info_parser_.getCameraInfo(frame_id_);
        } else {
            ROS_WARN("Failed to load camera intrinsics: %s", camera_info_url_.c_str());
        }
    }

    // Use default values if intrinsics are invalid
    if (!camera_info_parser_.isValid()) {
        camera_info_parser_.setImageSize(width_, height_);
        camera_info_msg_ = camera_info_parser_.getCameraInfo(frame_id_);
    }

    // Setup publishers
    setupPublishers();

    // Setup dynamic_reconfigure
    reconfigure_server_ = std::make_unique<dynamic_reconfigure::Server<CameraParamsConfig>>(pnh_);
    reconfigure_server_->setCallback(
        boost::bind(&CameraNode::reconfigureCallback, this, _1, _2));

    // Setup diagnostics
    diagnostics_.setHardwareID(device_);
    diagnostics_.add("Camera Status", this, &CameraNode::diagnosticCallback);

    ROS_INFO("Camera node initialized: %s", device_.c_str());
    return true;
}

void CameraNode::loadParameters() {
    pnh_.param<std::string>("device", device_, "/dev/video4");
    pnh_.param<std::string>("frame_id", frame_id_, "camera");
    pnh_.param<std::string>("camera_info_url", camera_info_url_, "");
    pnh_.param<int>("width", width_, 640);
    pnh_.param<int>("height", height_, 400);
    pnh_.param<double>("fps", fps_, 240.0);
    pnh_.param<int>("buffer_count", buffer_count_, 2);
    pnh_.param<int>("cpu_core", cpu_core_, -1);
    pnh_.param<int>("realtime_priority", realtime_priority_, 0);

    ROS_INFO("Parameters: device=%s, %dx%d@%.1ffps, frame_id=%s",
             device_.c_str(), width_, height_, fps_, frame_id_.c_str());
    ROS_INFO("Buffer count: %d, CPU core: %d, realtime priority: %d", 
             buffer_count_, cpu_core_, realtime_priority_);
}

void CameraNode::setupPublishers() {
    // Image + CameraInfo (image_transport will auto-create compressed topics)
    pub_camera_ = it_.advertiseCamera("image_raw", 1);
}

void CameraNode::reconfigureCallback(CameraParamsConfig& config, uint32_t level) {
    ROS_INFO("Received dynamic_reconfigure parameter update");
    current_config_ = config;
    applyConfig(config);
}

void CameraNode::applyConfig(const CameraParamsConfig& config) {
    if (!camera_ || !camera_->isOpen()) {
        return;
    }

    // Trigger mode
    camera_->setTriggerMode(config.trigger_mode == 1);

    // Exposure
    camera_->setControl(V4L2_CID_EXPOSURE_AUTO, 
                        config.exposure_auto ? V4L2_EXPOSURE_AUTO : V4L2_EXPOSURE_MANUAL);
    if (!config.exposure_auto) {
        camera_->setControl(V4L2_CID_EXPOSURE_ABSOLUTE, config.exposure_absolute);
    }

    // White balance
    camera_->setControl(V4L2_CID_AUTO_WHITE_BALANCE, config.white_balance_auto ? 1 : 0);
    if (!config.white_balance_auto) {
        camera_->setControl(V4L2_CID_WHITE_BALANCE_TEMPERATURE, 
                           config.white_balance_temperature);
    }

    // Image adjustments
    camera_->setControl(V4L2_CID_BRIGHTNESS, config.brightness);
    camera_->setControl(V4L2_CID_CONTRAST, config.contrast);
    camera_->setControl(V4L2_CID_SATURATION, config.saturation);
    camera_->setControl(V4L2_CID_HUE, config.hue);
    camera_->setControl(V4L2_CID_GAMMA, config.gamma);
    camera_->setControl(V4L2_CID_SHARPNESS, config.sharpness);
    camera_->setControl(V4L2_CID_BACKLIGHT_COMPENSATION, config.backlight_compensation);

    // Power line frequency
    camera_->setControl(V4L2_CID_POWER_LINE_FREQUENCY, config.power_line_frequency);
}

void CameraNode::spin() {
    if (!camera_->startStreaming()) {
        ROS_ERROR("Failed to start video stream");
        return;
    }

    running_.store(true);

    // Create capture thread
    capture_thread_ = std::make_unique<CameraThread>(
        "cam_capture", cpu_core_, realtime_priority_);

    capture_thread_->start([this]() {
        captureLoop();
    });

    // Main loop for ROS callbacks
    ros::Rate rate(100);  // 100Hz check
    while (ros::ok() && running_.load()) {
        ros::spinOnce();
        diagnostics_.update();
        rate.sleep();
    }

    stop();
}

void CameraNode::captureLoop() {
    int epoll_fd = epoll_create1(0);
    if (epoll_fd < 0) {
        ROS_ERROR("Failed to create epoll");
        return;
    }

    struct epoll_event ev;
    ev.events = EPOLLIN;
    ev.data.fd = camera_->getFd();
    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, ev.data.fd, &ev);

    struct epoll_event events[1];

    while (running_.load() && !capture_thread_->isStopRequested()) {
        int nfds = epoll_wait(epoll_fd, events, 1, 100);  // 100ms timeout

        if (nfds > 0) {
            FrameData frame;
            if (camera_->grabFrameNonBlocking(frame)) {
                ros::Time stamp = ros::Time::now();
                processAndPublish(frame, stamp);
                camera_->releaseFrame(frame.buffer_index);
                frame_count_++;
                last_frame_time_ = stamp;
            }
        }
    }

    close(epoll_fd);
}

void CameraNode::processAndPublish(const FrameData& frame, const ros::Time& stamp) {
    // Decode and publish Image (image_transport will auto-handle compression)
    if (jpeg_decoder_->decode(static_cast<const uint8_t*>(frame.data), frame.size,
                              rgb_buffer_.data(), rgb_buffer_.size(),
                              width_, height_)) {
        publishImage(rgb_buffer_.data(), width_, height_, stamp);
    } else {
        drop_count_++;
    }
}

void CameraNode::publishImage(const uint8_t* rgb_data, int width, int height,
                              const ros::Time& stamp) {
    // Apply time offset for trigger calibration
    ros::Time adjusted_stamp = stamp + ros::Duration(current_config_.time_offset);
    
    sensor_msgs::Image msg;
    msg.header.stamp = adjusted_stamp;
    msg.header.frame_id = frame_id_;
    msg.width = width;
    msg.height = height;
    msg.encoding = sensor_msgs::image_encodings::RGB8;
    msg.step = width * 3;
    msg.data.assign(rgb_data, rgb_data + width * height * 3);

    // Use CameraPublisher to publish Image and CameraInfo together
    sensor_msgs::CameraInfo info = camera_info_msg_;
    info.header = msg.header;
    pub_camera_.publish(msg, info);
}



void CameraNode::diagnosticCallback(diagnostic_updater::DiagnosticStatusWrapper& stat) {
    if (!camera_ || !camera_->isOpen()) {
        stat.summary(diagnostic_msgs::DiagnosticStatus::ERROR, "Camera not opened");
        return;
    }

    stat.summary(diagnostic_msgs::DiagnosticStatus::OK, "Camera OK");
    stat.add("Device", device_);
    stat.add("Resolution", std::to_string(width_) + "x" + std::to_string(height_));
    stat.add("Memory Type", camera_->getMemoryType() == MemoryType::DMABUF ? 
             "DMABUF" : "MMAP");
    stat.add("Frame Count", frame_count_.load());
    stat.add("Drop Count", drop_count_.load());

    if (last_frame_time_.toSec() > 0) {
        double age = (ros::Time::now() - last_frame_time_).toSec();
        stat.add("Frame Age (sec)", age);
    }
}

void CameraNode::stop() {
    running_.store(false);

    if (capture_thread_) {
        capture_thread_->stop();
        capture_thread_.reset();
    }

    if (camera_) {
        camera_->stopStreaming();
    }
}

// ============================================================================
// MultiCameraNode Implementation (Multi-camera Sync)
// ============================================================================

MultiCameraNode::MultiCameraNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
    : nh_(nh)
    , pnh_(pnh)
    , running_(false) {
}

MultiCameraNode::~MultiCameraNode() {
    stop();
}

bool MultiCameraNode::init() {
    loadParameters();

    // Create JPEG decoder (shared)
    jpeg_decoder_ = createBestJpegDecoder();
    if (!jpeg_decoder_) {
        ROS_ERROR("Failed to create JPEG decoder");
        return false;
    }

    // Initialize all cameras
    XmlRpc::XmlRpcValue cameras_param;
    if (!pnh_.getParam("cameras", cameras_param)) {
        ROS_ERROR("cameras parameter not found");
        return false;
    }

    for (auto it = cameras_param.begin(); it != cameras_param.end(); ++it) {
        std::string camera_id = it->first;
        XmlRpc::XmlRpcValue& cam_config = it->second;

        CameraResource resource;

        // Read parameters
        std::string device = static_cast<std::string>(cam_config["device"]);
        std::string frame_id = camera_id;
        if (cam_config.hasMember("frame_id")) {
            frame_id = static_cast<std::string>(cam_config["frame_id"]);
        }
        resource.frame_id = frame_id;

        int width = cam_config.hasMember("width") ? 
                    static_cast<int>(cam_config["width"]) : 640;
        int height = cam_config.hasMember("height") ? 
                     static_cast<int>(cam_config["height"]) : 400;
        int buffer_count = cam_config.hasMember("buffer_count") ?
                          static_cast<int>(cam_config["buffer_count"]) : 2;

        // Create camera
        resource.camera = std::make_unique<V4L2Camera>();
        if (!resource.camera->open(device)) {
            ROS_ERROR("Failed to open camera %s: %s", camera_id.c_str(), device.c_str());
            return false;
        }

        if (!resource.camera->setFormat(width, height, V4L2_PIX_FMT_MJPEG)) {
            ROS_ERROR("Failed to set format for camera %s", camera_id.c_str());
            return false;
        }

        if (!resource.camera->initBuffers(buffer_count)) {
            ROS_ERROR("Failed to initialize buffers for camera %s", camera_id.c_str());
            return false;
        }

        // Load intrinsics
        if (cam_config.hasMember("camera_info_url")) {
            std::string url = static_cast<std::string>(cam_config["camera_info_url"]);
            if (resource.camera_info_parser.loadOST(url)) {
                resource.camera_info_msg = resource.camera_info_parser.getCameraInfo(frame_id);
            }
        }

        // Create publishers (image_transport will auto-create compressed topics)
        ros::NodeHandle cam_nh(nh_, camera_id);
        image_transport::ImageTransport cam_it(cam_nh);
        resource.pub_camera = cam_it.advertiseCamera("image_raw", 1);

        // Register with sync manager
        sync_manager_.addCamera(camera_id, resource.camera.get());

        // Store resource first (need pointer for callback)
        camera_resources_[camera_id] = std::move(resource);
        
        // Setup dynamic_reconfigure for this camera
        ros::NodeHandle reconf_nh(pnh_, camera_id);
        camera_resources_[camera_id].reconfigure_server = 
            std::make_unique<dynamic_reconfigure::Server<CameraParamsConfig>>(reconf_nh);
        
        // Create callback with camera_id captured
        std::string cam_id_copy = camera_id;
        camera_resources_[camera_id].reconfigure_server->setCallback(
            [this, cam_id_copy](CameraParamsConfig& config, uint32_t level) {
                ROS_INFO("Received dynamic_reconfigure update for camera: %s", cam_id_copy.c_str());
                camera_resources_[cam_id_copy].current_config = config;
                applyConfig(cam_id_copy, config);
            });
        
        ROS_INFO("Initialized camera: %s (%s)", camera_id.c_str(), device.c_str());
    }

    // Allocate RGB buffer
    rgb_buffer_.resize(1920 * 1080 * 3);  // Max resolution

    ROS_INFO("Multi-camera node initialized, total %zu cameras", camera_resources_.size());
    return true;
}

void MultiCameraNode::applyConfig(const std::string& camera_id, const CameraParamsConfig& config) {
    auto it = camera_resources_.find(camera_id);
    if (it == camera_resources_.end()) {
        return;
    }

    V4L2Camera* camera = it->second.camera.get();
    if (!camera || !camera->isOpen()) {
        return;
    }

    // Trigger mode
    camera->setTriggerMode(config.trigger_mode == 1);

    // Exposure
    camera->setControl(V4L2_CID_EXPOSURE_AUTO, 
                       config.exposure_auto ? V4L2_EXPOSURE_AUTO : V4L2_EXPOSURE_MANUAL);
    if (!config.exposure_auto) {
        camera->setControl(V4L2_CID_EXPOSURE_ABSOLUTE, config.exposure_absolute);
    }

    // White balance
    camera->setControl(V4L2_CID_AUTO_WHITE_BALANCE, config.white_balance_auto ? 1 : 0);
    if (!config.white_balance_auto) {
        camera->setControl(V4L2_CID_WHITE_BALANCE_TEMPERATURE, 
                          config.white_balance_temperature);
    }

    // Image adjustments
    camera->setControl(V4L2_CID_BRIGHTNESS, config.brightness);
    camera->setControl(V4L2_CID_CONTRAST, config.contrast);
    camera->setControl(V4L2_CID_SATURATION, config.saturation);
    camera->setControl(V4L2_CID_HUE, config.hue);
    camera->setControl(V4L2_CID_GAMMA, config.gamma);
    camera->setControl(V4L2_CID_SHARPNESS, config.sharpness);
    camera->setControl(V4L2_CID_BACKLIGHT_COMPENSATION, config.backlight_compensation);

    // Power line frequency
    camera->setControl(V4L2_CID_POWER_LINE_FREQUENCY, config.power_line_frequency);
}

void MultiCameraNode::loadParameters() {
    // Multi-camera parameters are read in init()
}

void MultiCameraNode::spin() {
    if (!sync_manager_.startAllCameras()) {
        ROS_ERROR("Failed to start all cameras");
        return;
    }

    running_.store(true);

    // Create capture thread
    int cpu_core = -1;
    int realtime_priority = 0;
    pnh_.param<int>("cpu_core", cpu_core, -1);
    pnh_.param<int>("realtime_priority", realtime_priority, 0);

    capture_thread_ = std::make_unique<CameraThread>(
        "multi_cam", cpu_core, realtime_priority);

    capture_thread_->start([this]() {
        captureLoop();
    });

    // Main loop
    ros::Rate rate(100);
    while (ros::ok() && running_.load()) {
        ros::spinOnce();
        rate.sleep();
    }

    stop();
}

void MultiCameraNode::captureLoop() {
    while (running_.load() && !capture_thread_->isStopRequested()) {
        std::map<std::string, FrameData> frames;
        ros::Time stamp = sync_manager_.waitForSyncedFrames(frames, 100);

        if (stamp.toSec() == 0) {
            continue;  // Timeout or error
        }

        // Process each camera's frame
        for (auto& pair : frames) {
            const std::string& camera_id = pair.first;
            FrameData& frame = pair.second;

            auto res_it = camera_resources_.find(camera_id);
            if (res_it == camera_resources_.end()) {
                continue;
            }

            CameraResource& resource = res_it->second;
            int width = resource.camera->getWidth();
            int height = resource.camera->getHeight();

            // Decode and publish Image (image_transport will auto-handle compression)
            if (jpeg_decoder_->decode(static_cast<const uint8_t*>(frame.data), frame.size,
                                      rgb_buffer_.data(), rgb_buffer_.size(),
                                      width, height)) {
                // Apply time offset for trigger calibration
                ros::Time adjusted_stamp = stamp + 
                    ros::Duration(resource.current_config.time_offset);
                
                sensor_msgs::Image img_msg;
                img_msg.header.stamp = adjusted_stamp;
                img_msg.header.frame_id = resource.frame_id;
                img_msg.width = width;
                img_msg.height = height;
                img_msg.encoding = sensor_msgs::image_encodings::RGB8;
                img_msg.step = width * 3;
                img_msg.data.assign(rgb_buffer_.data(), 
                                   rgb_buffer_.data() + width * height * 3);

                sensor_msgs::CameraInfo info = resource.camera_info_msg;
                info.header = img_msg.header;
                resource.pub_camera.publish(img_msg, info);
            }

            // Release buffer
            resource.camera->releaseFrame(frame.buffer_index);
        }
    }
}

void MultiCameraNode::stop() {
    running_.store(false);

    if (capture_thread_) {
        capture_thread_->stop();
        capture_thread_.reset();
    }

    sync_manager_.stopAllCameras();
}

}  // namespace uvc_camera_driver

#ifndef UVC_CAMERA_DRIVER_CAMERA_NODE_H
#define UVC_CAMERA_DRIVER_CAMERA_NODE_H

#include "uvc_camera_driver/v4l2_camera.h"
#include "uvc_camera_driver/jpeg_decoder.h"
#include "uvc_camera_driver/camera_info_parser.h"
#include "uvc_camera_driver/camera_thread.h"
#include "uvc_camera_driver/sync_manager.h"
#include "uvc_camera_driver/CameraParamsConfig.h"

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/CameraInfo.h>
#include <dynamic_reconfigure/server.h>
#include <diagnostic_updater/diagnostic_updater.h>

#include <memory>
#include <atomic>

namespace uvc_camera_driver {

/**
 * @brief 单相机 ROS 节点类
 */
class CameraNode {
public:
    CameraNode(ros::NodeHandle& nh, ros::NodeHandle& pnh);
    ~CameraNode();

    /**
     * @brief 初始化节点
     * @return 成功返回 true
     */
    bool init();

    /**
     * @brief 主循环
     */
    void spin();

    /**
     * @brief 停止节点
     */
    void stop();

private:
    /**
     * @brief 加载 ROS 参数
     */
    void loadParameters();

    /**
     * @brief 配置发布者
     */
    void setupPublishers();

    /**
     * @brief dynamic_reconfigure 回调
     */
    void reconfigureCallback(CameraParamsConfig& config, uint32_t level);

    /**
     * @brief 应用相机控制参数
     */
    void applyConfig(const CameraParamsConfig& config);

    /**
     * @brief 相机读取线程任务
     */
    void captureLoop();

    /**
     * @brief 处理帧数据并发布
     */
    void processAndPublish(const FrameData& frame, const ros::Time& stamp);

    /**
     * @brief Publish Image message (decoded)
     */
    void publishImage(const uint8_t* rgb_data, int width, int height, 
                     const ros::Time& stamp);

    /**
     * @brief 诊断更新回调
     */
    void diagnosticCallback(diagnostic_updater::DiagnosticStatusWrapper& stat);

    // ROS 节点句柄
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    // Publishers
    image_transport::ImageTransport it_;
    image_transport::CameraPublisher pub_camera_;  // Image + CameraInfo

    // dynamic_reconfigure
    std::unique_ptr<dynamic_reconfigure::Server<CameraParamsConfig>> reconfigure_server_;
    CameraParamsConfig current_config_;

    // 诊断
    diagnostic_updater::Updater diagnostics_;

    // V4L2 相机
    std::unique_ptr<V4L2Camera> camera_;

    // JPEG 解码器
    std::unique_ptr<JpegDecoder> jpeg_decoder_;

    // 相机内参
    CameraInfoParser camera_info_parser_;
    sensor_msgs::CameraInfo camera_info_msg_;

    // 相机读取线程
    std::unique_ptr<CameraThread> capture_thread_;

    // 解码缓冲区
    std::vector<uint8_t> rgb_buffer_;

    // 参数
    std::string device_;
    std::string frame_id_;
    std::string camera_info_url_;
    int width_;
    int height_;
    double fps_;
    int buffer_count_;
    int cpu_core_;
    int realtime_priority_;

    // 状态
    std::atomic<bool> running_;
    std::atomic<uint64_t> frame_count_;
    std::atomic<uint64_t> drop_count_;
    ros::Time last_frame_time_;
};

/**
 * @brief Multi-camera sync ROS node class
 */
class MultiCameraNode {
public:
    MultiCameraNode(ros::NodeHandle& nh, ros::NodeHandle& pnh);
    ~MultiCameraNode();

    bool init();
    void spin();
    void stop();

private:
    void loadParameters();
    void captureLoop();
    
    /**
     * @brief Apply config to a specific camera
     */
    void applyConfig(const std::string& camera_id, const CameraParamsConfig& config);

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    // Sync manager
    SyncManager sync_manager_;

    // Per-camera resources
    struct CameraResource {
        std::unique_ptr<V4L2Camera> camera;
        image_transport::CameraPublisher pub_camera;
        CameraInfoParser camera_info_parser;
        sensor_msgs::CameraInfo camera_info_msg;
        std::string frame_id;
        std::unique_ptr<dynamic_reconfigure::Server<CameraParamsConfig>> reconfigure_server;
        CameraParamsConfig current_config;
    };
    std::map<std::string, CameraResource> camera_resources_;

    // JPEG decoder (shared)
    std::unique_ptr<JpegDecoder> jpeg_decoder_;

    // Capture thread
    std::unique_ptr<CameraThread> capture_thread_;

    // Decode buffer
    std::vector<uint8_t> rgb_buffer_;

    // State
    std::atomic<bool> running_;
};

}  // namespace uvc_camera_driver

#endif  // UVC_CAMERA_DRIVER_CAMERA_NODE_H

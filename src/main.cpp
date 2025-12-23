#include <ros/ros.h>
#include "uvc_camera_driver/camera_node.h"

#include <signal.h>

std::unique_ptr<uvc_camera_driver::CameraNode> g_camera_node;
std::unique_ptr<uvc_camera_driver::MultiCameraNode> g_multi_camera_node;

void signalHandler(int sig) {
    ROS_INFO("Received signal %d, shutting down...", sig);
    
    if (g_camera_node) {
        g_camera_node->stop();
    }
    if (g_multi_camera_node) {
        g_multi_camera_node->stop();
    }
    
    ros::shutdown();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "uvc_camera_node", ros::init_options::NoSigintHandler);
    
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // Check if multi-camera mode
    bool multi_camera = false;
    pnh.param<bool>("multi_camera", multi_camera, false);

    if (multi_camera) {
        ROS_INFO("Starting multi-camera sync mode");
        g_multi_camera_node = std::make_unique<uvc_camera_driver::MultiCameraNode>(nh, pnh);
        
        if (!g_multi_camera_node->init()) {
            ROS_ERROR("Multi-camera node initialization failed");
            return 1;
        }
        
        g_multi_camera_node->spin();
    } else {
        ROS_INFO("Starting single camera mode");
        g_camera_node = std::make_unique<uvc_camera_driver::CameraNode>(nh, pnh);
        
        if (!g_camera_node->init()) {
            ROS_ERROR("Camera node initialization failed");
            return 1;
        }
        
        g_camera_node->spin();
    }

    return 0;
}

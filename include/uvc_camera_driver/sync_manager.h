#ifndef UVC_CAMERA_DRIVER_SYNC_MANAGER_H
#define UVC_CAMERA_DRIVER_SYNC_MANAGER_H

#include "uvc_camera_driver/v4l2_camera.h"
#include <ros/ros.h>
#include <map>
#include <mutex>
#include <condition_variable>
#include <sys/epoll.h>

namespace uvc_camera_driver {

/**
 * @brief 多相机同步管理器
 * 
 * 使用 epoll 非阻塞等待多个相机帧，确保时间戳对齐
 */
class SyncManager {
public:
    SyncManager();
    ~SyncManager();

    // 禁用拷贝
    SyncManager(const SyncManager&) = delete;
    SyncManager& operator=(const SyncManager&) = delete;

    /**
     * @brief 注册相机
     * @param camera_id 相机标识符
     * @param camera 相机指针
     * @return 成功返回 true
     */
    bool addCamera(const std::string& camera_id, V4L2Camera* camera);

    /**
     * @brief 移除相机
     * @param camera_id 相机标识符
     */
    void removeCamera(const std::string& camera_id);

    /**
     * @brief 启动所有相机
     * @return 成功返回 true
     */
    bool startAllCameras();

    /**
     * @brief 停止所有相机
     */
    void stopAllCameras();

    /**
     * @brief 等待同步帧到达
     * 
     * 使用 epoll 等待所有相机的帧到达，返回统一的 ROS 时间戳
     * 
     * @param frames 输出帧数据映射 (camera_id -> FrameData)
     * @param timeout_ms 超时时间 (毫秒)
     * @return 成功返回所有帧的统一时间戳，失败返回零时间戳
     */
    ros::Time waitForSyncedFrames(std::map<std::string, FrameData>& frames,
                                  int timeout_ms = 100);

    /**
     * @brief 设置所有相机的触发模式
     * @param external true=外触发, false=视频流
     */
    void setTriggerModeAll(bool external);

    /**
     * @brief 获取相机数量
     */
    size_t getCameraCount() const;

    /**
     * @brief 获取相机 ID 列表
     */
    std::vector<std::string> getCameraIds() const;

private:
    /**
     * @brief 更新 epoll 监控
     */
    bool updateEpoll();

    int epoll_fd_;
    std::map<std::string, V4L2Camera*> cameras_;
    std::map<int, std::string> fd_to_camera_id_;  // fd -> camera_id 映射
    mutable std::mutex mutex_;
};

}  // namespace uvc_camera_driver

#endif  // UVC_CAMERA_DRIVER_SYNC_MANAGER_H

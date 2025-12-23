#include "uvc_camera_driver/sync_manager.h"

#include <unistd.h>
#include <cstring>
#include <ros/ros.h>

namespace uvc_camera_driver {

SyncManager::SyncManager()
    : epoll_fd_(-1) {
    epoll_fd_ = epoll_create1(0);
    if (epoll_fd_ < 0) {
        ROS_ERROR("Failed to create epoll instance: %s", strerror(errno));
    }
}

SyncManager::~SyncManager() {
    stopAllCameras();

    if (epoll_fd_ >= 0) {
        close(epoll_fd_);
        epoll_fd_ = -1;
    }
}

bool SyncManager::addCamera(const std::string& camera_id, V4L2Camera* camera) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!camera || !camera->isOpen()) {
        ROS_ERROR("Invalid camera object: %s", camera_id.c_str());
        return false;
    }

    cameras_[camera_id] = camera;
    fd_to_camera_id_[camera->getFd()] = camera_id;

    ROS_INFO("Registered camera: %s (fd=%d)", camera_id.c_str(), camera->getFd());
    return true;
}

void SyncManager::removeCamera(const std::string& camera_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = cameras_.find(camera_id);
    if (it != cameras_.end()) {
        int fd = it->second->getFd();
        
        // Remove from epoll
        if (epoll_fd_ >= 0) {
            epoll_ctl(epoll_fd_, EPOLL_CTL_DEL, fd, nullptr);
        }
        
        fd_to_camera_id_.erase(fd);
        cameras_.erase(it);
        
        ROS_INFO("Removed camera: %s", camera_id.c_str());
    }
}

bool SyncManager::startAllCameras() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& pair : cameras_) {
        if (!pair.second->startStreaming()) {
            ROS_ERROR("Failed to start camera %s", pair.first.c_str());
            return false;
        }
    }

    return updateEpoll();
}

void SyncManager::stopAllCameras() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& pair : cameras_) {
        pair.second->stopStreaming();
    }
}

bool SyncManager::updateEpoll() {
    if (epoll_fd_ < 0) {
        return false;
    }

    for (auto& pair : cameras_) {
        struct epoll_event ev;
        ev.events = EPOLLIN;
        ev.data.fd = pair.second->getFd();

        if (epoll_ctl(epoll_fd_, EPOLL_CTL_ADD, ev.data.fd, &ev) < 0) {
            if (errno != EEXIST) {
                ROS_ERROR("Failed to add fd to epoll: %s", strerror(errno));
                return false;
            }
        }
    }

    return true;
}

ros::Time SyncManager::waitForSyncedFrames(std::map<std::string, FrameData>& frames,
                                           int timeout_ms) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (cameras_.empty()) {
        return ros::Time(0);
    }

    frames.clear();
    
    const int max_events = cameras_.size();
    std::vector<struct epoll_event> events(max_events);
    std::set<std::string> received_cameras;

    ros::Time capture_time;
    bool first_frame = true;

    // Loop until all cameras have frames
    while (received_cameras.size() < cameras_.size()) {
        int nfds = epoll_wait(epoll_fd_, events.data(), max_events, timeout_ms);
        
        if (nfds < 0) {
            if (errno == EINTR) {
                continue;
            }
            ROS_ERROR("epoll_wait failed: %s", strerror(errno));
            return ros::Time(0);
        }

        if (nfds == 0) {
            // Timeout
            ROS_WARN("Timeout waiting for synced frames");
            return ros::Time(0);
        }

        for (int i = 0; i < nfds; ++i) {
            int fd = events[i].data.fd;
            auto it = fd_to_camera_id_.find(fd);
            if (it == fd_to_camera_id_.end()) {
                continue;
            }

            const std::string& camera_id = it->second;
            if (received_cameras.count(camera_id) > 0) {
                // Already have frame from this camera, skip
                continue;
            }

            V4L2Camera* camera = cameras_[camera_id];
            FrameData frame;

            if (camera->grabFrameNonBlocking(frame)) {
                // First frame determines timestamp
                if (first_frame) {
                    capture_time = ros::Time::now();
                    first_frame = false;
                }

                frames[camera_id] = frame;
                received_cameras.insert(camera_id);
            }
        }
    }

    return capture_time;
}

void SyncManager::setTriggerModeAll(bool external) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& pair : cameras_) {
        pair.second->setTriggerMode(external);
    }

    ROS_INFO("All cameras trigger mode set to: %s", external ? "External" : "Video Stream");
}

size_t SyncManager::getCameraCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cameras_.size();
}

std::vector<std::string> SyncManager::getCameraIds() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> ids;
    ids.reserve(cameras_.size());
    for (const auto& pair : cameras_) {
        ids.push_back(pair.first);
    }
    return ids;
}

}  // namespace uvc_camera_driver

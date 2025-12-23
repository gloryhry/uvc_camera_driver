#include "uvc_camera_driver/camera_thread.h"

#include <pthread.h>
#include <sched.h>
#include <cstring>
#include <ros/ros.h>

namespace uvc_camera_driver {

CameraThread::CameraThread(const std::string& name, int cpu_core, int realtime_priority)
    : name_(name)
    , cpu_core_(cpu_core)
    , realtime_priority_(realtime_priority)
    , running_(false)
    , stop_requested_(false) {
}

CameraThread::~CameraThread() {
    stop();
}

void CameraThread::start(std::function<void()> task) {
    if (running_.load()) {
        return;
    }

    stop_requested_.store(false);
    running_.store(true);

    thread_ = std::thread([this, task]() {
        // Configure thread properties
        configureThread();
        
        // Execute task
        task();
        
        running_.store(false);
    });
}

void CameraThread::stop() {
    stop_requested_.store(true);

    if (thread_.joinable()) {
        thread_.join();
    }

    running_.store(false);
}

void CameraThread::configureThread() {
    // Set thread name
    if (!name_.empty()) {
        pthread_setname_np(pthread_self(), name_.substr(0, 15).c_str());
    }

    // Set CPU affinity
    if (cpu_core_ >= 0) {
        if (setCpuAffinity(cpu_core_)) {
            ROS_INFO("Thread [%s] bound to CPU core %d", name_.c_str(), cpu_core_);
        } else {
            ROS_WARN("Thread [%s] failed to bind to CPU core %d", name_.c_str(), cpu_core_);
        }
    }

    // Set realtime priority
    if (realtime_priority_ > 0) {
        if (setRealtimePriority(realtime_priority_)) {
            ROS_INFO("Thread [%s] set realtime priority %d (SCHED_FIFO)", 
                     name_.c_str(), realtime_priority_);
        } else {
            ROS_WARN("Thread [%s] failed to set realtime priority %d (requires root)", 
                     name_.c_str(), realtime_priority_);
        }
    }
}

bool CameraThread::setCpuAffinity(int core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);

    int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        ROS_WARN("pthread_setaffinity_np failed: %s", strerror(result));
        return false;
    }

    return true;
}

bool CameraThread::setRealtimePriority(int priority) {
    if (priority < 1 || priority > 99) {
        ROS_WARN("Invalid realtime priority: %d (valid range 1-99)", priority);
        return false;
    }

    struct sched_param param;
    param.sched_priority = priority;

    int result = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
    if (result != 0) {
        ROS_WARN("pthread_setschedparam failed: %s", strerror(result));
        return false;
    }

    return true;
}

}  // namespace uvc_camera_driver

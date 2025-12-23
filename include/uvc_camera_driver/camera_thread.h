#ifndef UVC_CAMERA_DRIVER_CAMERA_THREAD_H
#define UVC_CAMERA_DRIVER_CAMERA_THREAD_H

#include <thread>
#include <functional>
#include <atomic>
#include <string>

namespace uvc_camera_driver {

/**
 * @brief 相机读取线程类
 * 
 * 支持 CPU 核心绑定和实时调度优先级
 */
class CameraThread {
public:
    /**
     * @brief 构造函数
     * @param name 线程名称 (用于调试)
     * @param cpu_core CPU 核心索引 (-1 表示不绑定)
     * @param realtime_priority 实时优先级 (0 表示不设置, 1-99 为 SCHED_FIFO)
     */
    CameraThread(const std::string& name = "", 
                 int cpu_core = -1, 
                 int realtime_priority = 0);

    ~CameraThread();

    // 禁用拷贝
    CameraThread(const CameraThread&) = delete;
    CameraThread& operator=(const CameraThread&) = delete;

    /**
     * @brief 启动线程
     * @param task 线程执行的任务函数
     */
    void start(std::function<void()> task);

    /**
     * @brief 停止线程
     */
    void stop();

    /**
     * @brief 检查线程是否正在运行
     */
    bool isRunning() const { return running_.load(); }

    /**
     * @brief 请求停止
     */
    void requestStop() { stop_requested_.store(true); }

    /**
     * @brief 检查是否请求停止
     */
    bool isStopRequested() const { return stop_requested_.load(); }

    /**
     * @brief 设置 CPU 亲和性
     * @param core CPU 核心索引
     * @return 成功返回 true
     */
    bool setCpuAffinity(int core);

    /**
     * @brief 设置实时调度优先级
     * @param priority 优先级 (1-99, SCHED_FIFO)
     * @return 成功返回 true
     */
    bool setRealtimePriority(int priority);

    /**
     * @brief 获取线程名称
     */
    const std::string& getName() const { return name_; }

private:
    /**
     * @brief 配置线程属性 (在线程内部调用)
     */
    void configureThread();

    std::string name_;
    int cpu_core_;
    int realtime_priority_;
    std::thread thread_;
    std::atomic<bool> running_;
    std::atomic<bool> stop_requested_;
};

}  // namespace uvc_camera_driver

#endif  // UVC_CAMERA_DRIVER_CAMERA_THREAD_H

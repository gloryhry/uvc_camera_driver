#ifndef UVC_CAMERA_DRIVER_CAMERA_INFO_PARSER_H
#define UVC_CAMERA_DRIVER_CAMERA_INFO_PARSER_H

#include <string>
#include <sensor_msgs/CameraInfo.h>

namespace uvc_camera_driver {

/**
 * @brief 相机内参解析器
 * 
 * 支持 oST v5.0 格式文件解析
 */
class CameraInfoParser {
public:
    CameraInfoParser();

    /**
     * @brief 加载 oST 格式相机内参文件
     * @param filepath 文件路径
     * @return 成功返回 true
     */
    bool loadOST(const std::string& filepath);

    /**
     * @brief 获取 ROS CameraInfo 消息
     * @param frame_id TF 坐标系名称
     * @return CameraInfo 消息
     */
    sensor_msgs::CameraInfo getCameraInfo(const std::string& frame_id = "") const;

    /**
     * @brief 设置图像尺寸 (如果文件中未指定)
     */
    void setImageSize(int width, int height);

    /**
     * @brief 检查是否已加载有效数据
     */
    bool isValid() const { return valid_; }

private:
    /**
     * @brief 解析 3x3 矩阵
     */
    bool parseMatrix3x3(std::istream& is, double* matrix);

    /**
     * @brief 解析 3x4 矩阵
     */
    bool parseMatrix3x4(std::istream& is, double* matrix);

    /**
     * @brief 解析畸变系数
     */
    bool parseDistortion(std::istream& is);

    bool valid_;
    int width_;
    int height_;
    double K_[9];    // camera matrix 3x3
    double D_[5];    // distortion coefficients
    double R_[9];    // rectification matrix 3x3
    double P_[12];   // projection matrix 3x4
};

}  // namespace uvc_camera_driver

#endif  // UVC_CAMERA_DRIVER_CAMERA_INFO_PARSER_H

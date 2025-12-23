#include "uvc_camera_driver/camera_info_parser.h"

#include <fstream>
#include <sstream>
#include <algorithm>
#include <ros/ros.h>

namespace uvc_camera_driver {

CameraInfoParser::CameraInfoParser()
    : valid_(false)
    , width_(0)
    , height_(0) {
    std::fill(K_, K_ + 9, 0.0);
    std::fill(D_, D_ + 5, 0.0);
    std::fill(R_, R_ + 9, 0.0);
    std::fill(P_, P_ + 12, 0.0);

    // Default identity matrix
    K_[0] = K_[4] = K_[8] = 1.0;
    R_[0] = R_[4] = R_[8] = 1.0;
}

bool CameraInfoParser::loadOST(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        ROS_ERROR("Failed to open camera intrinsics file: %s", filepath.c_str());
        return false;
    }

    std::string line;
    std::string current_section;

    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') {
            continue;
        }

        // Detect section
        if (line[0] == '[') {
            size_t end = line.find(']');
            if (end != std::string::npos) {
                current_section = line.substr(1, end - 1);
            }
            continue;
        }

        // Parse [image] section
        if (current_section == "image") {
            if (line == "width") {
                std::getline(file, line);
                width_ = std::stoi(line);
            } else if (line == "height") {
                std::getline(file, line);
                height_ = std::stoi(line);
            }
        }
        // Parse [narrow_stereo] section
        else if (current_section == "narrow_stereo") {
            if (line.find("camera matrix") != std::string::npos) {
                if (!parseMatrix3x3(file, K_)) {
                    ROS_WARN("Failed to parse camera matrix");
                }
            } else if (line.find("distortion") != std::string::npos) {
                if (!parseDistortion(file)) {
                    ROS_WARN("Failed to parse distortion");
                }
            } else if (line.find("rectification") != std::string::npos) {
                if (!parseMatrix3x3(file, R_)) {
                    ROS_WARN("Failed to parse rectification");
                }
            } else if (line.find("projection") != std::string::npos) {
                if (!parseMatrix3x4(file, P_)) {
                    ROS_WARN("Failed to parse projection");
                }
            }
        }
    }

    valid_ = (width_ > 0 && height_ > 0);
    if (valid_) {
        ROS_INFO("Loaded camera intrinsics: %dx%d", width_, height_);
    }

    return valid_;
}

bool CameraInfoParser::parseMatrix3x3(std::istream& is, double* matrix) {
    for (int row = 0; row < 3; ++row) {
        std::string line;
        if (!std::getline(is, line)) {
            return false;
        }
        
        std::istringstream iss(line);
        for (int col = 0; col < 3; ++col) {
            if (!(iss >> matrix[row * 3 + col])) {
                return false;
            }
        }
    }
    return true;
}

bool CameraInfoParser::parseMatrix3x4(std::istream& is, double* matrix) {
    for (int row = 0; row < 3; ++row) {
        std::string line;
        if (!std::getline(is, line)) {
            return false;
        }
        
        std::istringstream iss(line);
        for (int col = 0; col < 4; ++col) {
            if (!(iss >> matrix[row * 4 + col])) {
                return false;
            }
        }
    }
    return true;
}

bool CameraInfoParser::parseDistortion(std::istream& is) {
    std::string line;
    if (!std::getline(is, line)) {
        return false;
    }

    std::istringstream iss(line);
    int count = 0;
    double val;
    while (iss >> val && count < 5) {
        D_[count++] = val;
    }

    return count > 0;
}

void CameraInfoParser::setImageSize(int width, int height) {
    width_ = width;
    height_ = height;
}

sensor_msgs::CameraInfo CameraInfoParser::getCameraInfo(const std::string& frame_id) const {
    sensor_msgs::CameraInfo info;

    info.header.frame_id = frame_id;
    info.width = width_;
    info.height = height_;

    // Distortion model
    info.distortion_model = "plumb_bob";

    // Distortion coefficients D (5)
    info.D.resize(5);
    for (int i = 0; i < 5; ++i) {
        info.D[i] = D_[i];
    }

    // Camera matrix K (3x3, row-major)
    for (int i = 0; i < 9; ++i) {
        info.K[i] = K_[i];
    }

    // Rectification matrix R (3x3)
    for (int i = 0; i < 9; ++i) {
        info.R[i] = R_[i];
    }

    // Projection matrix P (3x4)
    for (int i = 0; i < 12; ++i) {
        info.P[i] = P_[i];
    }

    // binning
    info.binning_x = 1;
    info.binning_y = 1;

    // ROI (full image)
    info.roi.x_offset = 0;
    info.roi.y_offset = 0;
    info.roi.width = 0;
    info.roi.height = 0;
    info.roi.do_rectify = false;

    return info;
}

}  // namespace uvc_camera_driver

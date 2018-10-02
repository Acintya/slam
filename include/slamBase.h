
# pragma once

// 各种头文件 
// C++标准库
#include <fstream>
#include <vector>
using namespace std;

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// 类型定义
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

struct CAMERA_INTRINSIC_PARAMETERS
{
    double cx, cy, fx, fy, scale;
};

// function interfaces
// rgb -> point cloud
PointCloud::Ptr image2PointCloud (cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera);

// point2DTo3D
cv::Point3f point2DTo3D (cv::Point3f & point, CAMERA_INTRINSIC_PARAMETERS & camera, string pcdFilePath);
// save pc to pcd file
void savePC2PCD (PointCloud::Ptr cloudPtr, string pcdFilePath);
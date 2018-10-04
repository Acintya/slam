# pragma once

// 各种头文件 
// C++标准库
#include <fstream>
#include <vector>
using namespace std;

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
// openCV feature detection module
#include<opencv2/features2d/features2d.hpp>
//#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/calib3d/calib3d.hpp>

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

void calculateRandTwithPnP (
    cv::Mat depth1,
    vector<cv::KeyPoint> vecKP1,
    vector<cv::KeyPoint> vecKP2,
    vector<cv::DMatch> goodMatches,
    CAMERA_INTRINSIC_PARAMETERS camParams,
    cv::Mat& vecR,
    cv::Mat& vecT,
    cv::Mat& inliers);

// Frame
struct tFrame
{
    cv::Mat rgbImg, depthImg;
    cv::Mat desp;
    vector<cv::KeyPoint> vecKP;
};

// Result of Pnp
struct tResultOfPnP
{
    cv::Mat vecT, vecR, inliers;
    int numInliers;
};

// compute key points and descriptor
void computeKeyPointsAndDesp (tFrame& frame, string detector, string descriptor);

// estimate the motion between two frames
// Input: frame 1 and 2, cam params
// save result in tResultOfPnP
void estimateMotion (tFrame &, tFrame &, CAMERA_INTRINSIC_PARAMETERS, tResultOfPnP & _resultOfPnP);
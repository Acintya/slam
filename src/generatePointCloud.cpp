#include "slamBase.h"
#include "slamBase.cpp"

//main
int main (int argc, char** argv)
{
    cv::Mat rgb = cv::imread("../data/rgb.png");
    cout << rgb.rows;
    cv::Mat depth = cv::imread("../data/depth.png");

    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.fx = 518.0;
    camera.fy = 519.0;
    camera.scale = 1000;
    camera.cx = 325.5;
    camera.cy = 253.5;
    PointCloud::Ptr _pointCloudPtr;

    string pcdPath = "../data/newpcd.pcd";
    _pointCloudPtr = image2PointCloud(rgb, depth, camera, pcdPath);

}
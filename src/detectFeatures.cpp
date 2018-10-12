#include<iostream>
#include"slamBase.h"
#include"slamBase.cpp"
using namespace std;

// openCV feature detection module
#include <opencv2/features2d/features2d.hpp>
//#include<opencv2/nonfree/nonfree.hpp>
#include <opencv2/calib3d/calib3d.hpp>
//for laptop camera
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

int main (int argc, char** argv)
{
    //open cam
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        return -1;
    }
    cv::Mat frame, edges;
    bool stop = false;
    while (!stop)
    {
        cap >> frame;
        cv::imshow("cam", frame);
        if (cv::waitKey(30) >= 0)
            stop =true;
    }

    // read two rgb and depth img
    cv::Mat rgb1 = cv::imread("/home/ling/slam/data/rgb1.png");
    cv::Mat rgb2 = cv::imread("/home/ling/slam/data/rgb2.png");
    cv::Mat depth1 = cv::imread("/home/ling/slam/data/depth1.png", -1);
    cv::Mat depth2 = cv::imread("/home/ling/slam/data/depth2.png", -1);

    tFrame frame1;
    frame1.rgbImg = rgb1;
    frame1.depthImg = depth1;

    tFrame frame2;
    frame2.rgbImg = rgb2;
    frame2.depthImg = depth2;

    computeKeyPointsAndDesp(frame1, "ORB", "ORB");
    computeKeyPointsAndDesp(frame2, "ORB", "ORB");

    tResultOfPnP _resultOfPnP;

    //TO-DO: get camParam from paramReader
    CAMERA_INTRINSIC_PARAMETERS camParams;
    camParams.cx = 325.5;
    camParams.cy = 253.5;
    camParams.fx = 518.0;
    camParams.fy = 519.0;
    camParams.scale = 1000.0;
    estimateMotion (
        frame1, 
        frame2, 
        camParams, 
        _resultOfPnP);

    return 0;
}
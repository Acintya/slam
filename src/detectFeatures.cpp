#include<iostream>
#include"slamBase.h"
#include"slamBase.cpp"
using namespace std;

// openCV feature detection module
#include<opencv2/features2d/features2d.hpp>
//#include<opencv2/nonfree/nonfree.hpp>
#include<opencv2/calib3d/calib3d.hpp>

int main (int argc, char** argv)
{
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
    
    // // visu matches
    // cv::drawMatches(rgb1, vecKP1, 
    //     rgb2, vecKP2, 
    //     matches, imgShow);
    // cv::imshow("matches", imgShow);
    // cv::imwrite("../data/out/matches.png", imgShow);
    // cv::waitKey(0);

    // // visu the good matches
    // cout << "good matches = " << goodMatches.size() << endl;
    // cv::drawMatches(rgb1, vecKP1, rgb2, vecKP2, goodMatches, imgShow);
    // cv::imshow("good matches!", imgShow);
    // cv::imwrite("../data/out/god_matches.png", imgShow);
    // cv::waitKey(0); 

    return 0;
}
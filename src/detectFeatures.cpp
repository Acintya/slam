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

    // declare feature detector and descriptor
    cv::Ptr<cv::FeatureDetector> detector;
    cv::Ptr<cv::DescriptorExtractor> descriptor;

    // build detector and descriptor
    // for SIFT & SURF: init nonfree at beginning
    //cv::initModule_nonfree;
    detector = cv::FeatureDetector::create("ORB");
    descriptor = cv::DescriptorExtractor::create("ORB");

    // extract the keypoints of two imgs and save to vecKP1,2
    vector<cv::KeyPoint> vecKP1, vecKP2;
    detector->detect(rgb1, vecKP1);
    detector->detect(rgb2, vecKP2);

    cout << "key points found in rgb1: " << vecKP1.size();
    cout << endl;
    cout << "key points found in rgb2: " << vecKP2.size();
    cout << endl;

    // visu
    cv::Mat imgShow;
    cv::drawKeypoints(rgb1, vecKP1, imgShow, cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("keypoints", imgShow);
    cv::imwrite("../data/out/keypoints1.png", imgShow);
    cv::waitKey(0);

    cv::drawKeypoints(rgb2, vecKP2, imgShow, cv::Scalar::all(-1),
        cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("keypoints2", imgShow);
    cv::imwrite("../data/out/keypoints2.png", imgShow);
    cv::waitKey(0);

    // compute descriptor
    cv::Mat despMat1, despMat2;
    descriptor->compute(rgb1, vecKP1, despMat1);
    descriptor->compute(rgb2, vecKP2, despMat2);

    // register the descriptor
    vector<cv::DMatch> matches;
    cv::BFMatcher matcher;
    matcher.match(despMat1, despMat2, matches);
    cout << "find total " << matches.size() << "matches." << endl;

    // visu matches
    cv::drawMatches(rgb1, vecKP1, 
        rgb2, vecKP2, 
        matches, imgShow);
    cv::imshow("matches", imgShow);
    cv::imwrite("../data/out/matches.png", imgShow);
    cv::waitKey(0);

    // filter out the points with large distance
    // rule here: point.distance > 4 * min_dis is out!
    vector<cv::DMatch> goodMatches;
    double minDistance = 9999;
    
    for(size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i].distance < minDistance)
            minDistance = matches[i].distance;
    }
    
    for(size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i].distance < 8 * minDistance)
            goodMatches.push_back(matches[i]);
    }

    // visu the good matches
    cout << "good matches = " << goodMatches.size() << endl;
    cv::drawMatches(rgb1, vecKP1, rgb2, vecKP2, goodMatches, imgShow);
    cv::imshow("good matches!", imgShow);
    cv::imwrite("../data/out/god_matches.png", imgShow);
    cv::waitKey(0);

    // camera params
    CAMERA_INTRINSIC_PARAMETERS camParams;
    camParams.cx = 325.5;
    camParams.cy = 253.5;
    camParams.fx = 518.0;
    camParams.fy = 519.0;
    camParams.scale = 1000.0;

//////
    cv::Mat vecR, vecT, inliers;
    calculateRandTwithPnP(
        depth1,
        vecKP1,
        vecKP2,
        goodMatches, 
        camParams, 
        vecR, vecT, inliers);
    cout << "inliers" << inliers.rows << endl;
    cout << "R = " << vecR << endl;
    cout << "T = " << vecT << endl;

    // visu inliers
    vector<cv::DMatch> matchesShow;
    
    for(size_t i = 0; i < inliers.rows; i++)
    {
        matchesShow.push_back(goodMatches[inliers.ptr<int>(i)[0]]);
    }
    cv::drawMatches(rgb1, vecKP1, rgb2, vecKP2, matchesShow, imgShow);
    cv::imshow("show inliers", imgShow);
    cv::imwrite("../data/out/inliers.png", imgShow);
    cv::waitKey(0);

    return 0;
}
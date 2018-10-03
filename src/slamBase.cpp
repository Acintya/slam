#include "slamBase.h"

PointCloud::Ptr image2PointCloud( cv::Mat& rgb, 
    cv::Mat& depth, 
    CAMERA_INTRINSIC_PARAMETERS& camera,
    string pcdFilePath)
{
    PointCloud::Ptr cloud ( new PointCloud );

    for (int m = 0; m < depth.rows; m++)
        for (int n=0; n < depth.cols; n++)
        {
            // 获取深度图中(m,n)处的值
            ushort d = depth.ptr<ushort>(m)[n];
            // d 可能没有值，若如此，跳过此点
            if (d == 0)
                continue;
            // d 存在值，则向点云增加一个点
            PointT p;

            // 计算这个点的空间坐标
            p.z = double(d) / camera.scale;
            p.x = (n - camera.cx) * p.z / camera.fx;
            p.y = (m - camera.cy) * p.z / camera.fy;
            
            // 从rgb图像中获取它的颜色
            // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            // 把p加入到点云中
            cloud->points.push_back( p );
        }
    // 设置并保存点云
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;

    pcl::io::savePCDFile(pcdFilePath, *cloud);
    cout << "Point cloud saved." << endl;
    // clean up the data and quit
    cloud->points.clear();

    return cloud;
}

void savePC2PCD (PointCloud::Ptr cloudPtr, string pcdFilePath)
{
    pcl::io::savePCDFile(pcdFilePath, *cloudPtr);
}

cv::Point3f point2dTo3d( cv::Point3f& point, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    cv::Point3f p; // 3D 点
    p.z = double( point.z ) / camera.scale;
    p.x = ( point.x - camera.cx) * p.z / camera.fx;
    p.y = ( point.y - camera.cy) * p.z / camera.fy;
    return p;
}

// calculate the motion between two imgs
// key func: cv::solvePnPRansac()
void calculateRandTwithPnP (
    cv::Mat depth1,
    vector<cv::KeyPoint> vecKP1,
    vector<cv::KeyPoint> vecKP2,
    vector<cv::DMatch> goodMatches,
    CAMERA_INTRINSIC_PARAMETERS camParams,
    cv::Mat& vecR,
    cv::Mat& vecT,
    cv::Mat& inliers)
{
    // set up the params for the function

    // the 3d point from the first img
    vector<cv::Point3f> pts_obj;

    // the 2d point on the second img
    vector<cv::Point2f> pts_img;

    for(size_t i = 0; i < goodMatches.size(); i++)
    {
        //query: rgb1, train: rgb2
        cv::Point2f p = vecKP1[goodMatches[i].queryIdx].pt;
        // y x ->
        // |
        // v
        ushort d = depth1.ptr<ushort>(int(p.y))[int(p.x)];
        // ignore if d == 0
        if (d == 0)
            continue;
        pts_img.push_back(cv::Point2f(vecKP2[goodMatches[i].trainIdx].pt));

        // (u, v, d) -> (x, y, z)
        cv::Point3f pt (p.x, p.y, d);
        cv::Point3f pd = point2dTo3d(pt, camParams);
        pts_obj.push_back(pd);
    }
    
    double camera_matrix_data[3][3] = {
        {camParams.fx, 0, camParams.cx},
        {0, camParams.fy, camParams.cy},
        {0, 0, 1}
    };

    // generate camera matrix
    cv::Mat cameraMatrix(3, 3, CV_64F, camera_matrix_data);
    //cv::Mat vecR, vecT, inliers;
    // solve PnP
    cv::solvePnPRansac(
        pts_obj,
        pts_img,
        cameraMatrix,
        cv::Mat(),
        vecR,
        vecT,
        false,
        100,
        1.0,
        100,
        inliers
    );
}
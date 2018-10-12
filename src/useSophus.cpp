#include <iostream>
#include <cmath>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "sophus/so3.h"
#include "sophus/se3.h"

int main(int argc, char const *argv[])
{
    //90 degree around z
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();

    Sophus::SO3 SO3_R(R); // SO3 can constructed from rotation matrix directly
    Sophus::SO3 SO3_v(0, 0, M_PI / 2); // or from rotation vector
    Eigen::Quaterniond q(R); // or quaternion
    Sophus::SO3 SO3_q(q);

    // output
    cout << "SO(3) from matrix: " << SO3_R << endl;
    cout << "SO(3) from vector: " << SO3_v << endl;
    cout << "SO(3) from quaternion: " << SO3_q << endl;

    return 0;
}

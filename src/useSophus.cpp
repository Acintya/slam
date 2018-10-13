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

    // get lie algebra via exponential map
    Eigen::Vector3d so3 = SO3_R.log();
    cout << "so3 = " << so3.transpose() << endl;
    // hat = inverse-symmetrisic matrix of vector
    cout << "so3_hat = " << endl << Sophus::SO3::hat(so3) << endl;
    // vee = inverse-symmetrisic vector of matrix
    cout << "so3_vee = " << endl << Sophus::SO3::vee(Sophus::SO3::hat(so3)).transpose() << endl;

    // small noise model
    Eigen::Vector3d update_so3 (1e-4, 0, 0);
    Sophus::SO3 SO3_updated = Sophus::SO3::exp(update_so3) * SO3_R;
    cout << "SO3_updated =" << endl;
    cout << SO3_updated << endl;

    /********SE(3)***********/
    Eigen::Vector3d t (1, 0, 0); // translation along x axis
    Sophus::SE3 SE3_Rt(R, t); // generate SE(3) with R, t
    Sophus::SE3 SE3_qt(q, t);
    cout << "SE_Rt = " << endl << SE3_Rt << endl; 
    cout << "SE_qt = " << endl << SE3_qt << endl;

    //se(3)
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    Vector6d se3 = SE3_qt.log();
    cout << "se(3) = " << endl << se3 << endl;

    cout << "se(3) hat:" << Sophus::SE3::hat(se3) << endl;
    cout << "se(3) matrix vee" << Sophus::SE3::vee(Sophus::SE3::hat(se3)).transpose() << endl;
    
    Vector6d se3_update;
    se3_update.setZero();
    Sophus::SE3 SE3_updated = Sophus::SE3::exp(se3_update)*SE3_Rt;
    cout << "SE_3updated: " << SE3_updated.matrix() << endl;
    return 0;
}

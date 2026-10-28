#pragma once

#include <Eigen/Core>

namespace litepgo {

// Numerical shell inputs. Reference geometry, thickness, and parameters are held
// fixed when differentiating the current symmetric forms (a11,a12,a22,b11,b12,b22).
struct ShellMaterialInputs {
    Eigen::Matrix2d a;
    Eigen::Matrix2d b;
    Eigen::Matrix2d a_bar;
    Eigen::Matrix2d b_bar;
    double thickness;
    Eigen::VectorXd theta;
};

} // namespace litepgo

#pragma once
#include <Eigen/Dense>

struct CartPoleParams {
    double M = 1.0;   // cart mass (kg)
    double m = 0.1;   // pole mass (kg)
    double l = 0.5;   // pole COM length (m)
    double g = 9.81;  // gravity (m/s^2)
    double b = 0.0;   // cart viscous friction (N*s/m)
};

// State x = [p, p_dot, theta, theta_dot]^T
// theta = 0 means upright
Eigen::Vector4d cartpole_dynamics(const Eigen::Vector4d& x,
                                 double u,
                                 const CartPoleParams& p);

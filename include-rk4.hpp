#pragma once
#include <Eigen/Dense>

template <typename DynamicsFunc, typename Params>
Eigen::Vector4d rk4_step(DynamicsFunc f,
                         const Eigen::Vector4d& x,
                         double u,
                         double dt,
                         const Params& prm)
{
    const Eigen::Vector4d k1 = f(x, u, prm);
    const Eigen::Vector4d k2 = f(x + 0.5 * dt * k1, u, prm);
    const Eigen::Vector4d k3 = f(x + 0.5 * dt * k2, u, prm);
    const Eigen::Vector4d k4 = f(x + dt * k3, u, prm);

    return x + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
}

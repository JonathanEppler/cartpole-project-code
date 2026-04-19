#pragma once
#include <Eigen/Dense>
#include "cartpole.hpp"
#include "rk4.hpp"

class LQRController {
public:
    LQRController() {
        K_.setZero();
    }

    void set_gain(const Eigen::Matrix<double, 1, 4>& K) {
        K_ = K;
    }

    const Eigen::Matrix<double, 1, 4>& gain() const {
        return K_;
    }

    double control(const Eigen::Vector4d& x_est) const {
        Eigen::Matrix<double, 1, 1> u = -K_ * x_est;
        return u(0, 0);
    }

private:
    Eigen::Matrix<double, 1, 4> K_;
};

inline Eigen::Matrix<double, 1, 4> compute_discrete_lqr_gain(
    const Eigen::Matrix4d& A,
    const Eigen::Matrix<double, 4, 1>& B,
    const Eigen::Matrix4d& Q,
    const Eigen::Matrix<double, 1, 1>& R,
    int max_iters = 1000,
    double tol = 1e-9)
{
    Eigen::Matrix4d P = Q;

    for (int i = 0; i < max_iters; ++i) {
        Eigen::Matrix<double, 1, 1> S = R + B.transpose() * P * B;
        Eigen::Matrix<double, 1, 4> K = S.inverse() * (B.transpose() * P * A);

        Eigen::Matrix4d P_next =
            A.transpose() * P * A
            - A.transpose() * P * B * K
            + Q;

        if ((P_next - P).norm() < tol) {
            P = P_next;
            break;
        }

        P = P_next;
    }

    Eigen::Matrix<double, 1, 1> S = R + B.transpose() * P * B;
    Eigen::Matrix<double, 1, 4> K = S.inverse() * (B.transpose() * P * A);

    return K;
}

inline Eigen::Vector4d discrete_step(
    const Eigen::Vector4d& x,
    double u,
    double dt,
    const CartPoleParams& prm)
{
    return rk4_step(cartpole_dynamics, x, u, dt, prm);
}

inline void get_numerical_discrete_cartpole_linearization(
    double dt,
    const CartPoleParams& prm,
    Eigen::Matrix4d& A_d,
    Eigen::Matrix<double, 4, 1>& B_d)
{
    const double eps_x = 1e-6;
    const double eps_u = 1e-6;

    Eigen::Vector4d x0 = Eigen::Vector4d::Zero();
    double u0 = 0.0;

    Eigen::Vector4d f0 = discrete_step(x0, u0, dt, prm);

    A_d.setZero();
    B_d.setZero();

    for (int i = 0; i < 4; ++i) {
        Eigen::Vector4d x_pert = x0;
        x_pert(i) += eps_x;

        Eigen::Vector4d fi = discrete_step(x_pert, u0, dt, prm);
        A_d.col(i) = (fi - f0) / eps_x;
    }

    {
        Eigen::Vector4d fu = discrete_step(x0, u0 + eps_u, dt, prm);
        B_d = (fu - f0) / eps_u;
    }
}

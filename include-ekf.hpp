#pragma once
#include <Eigen/Dense>
#include "cartpole.hpp"
#include "rk4.hpp"

class ExtendedKalmanFilter {
public:
    ExtendedKalmanFilter() {
        x_hat_.setZero();
        P_.setIdentity();
        Q_.setIdentity();
        R_.setIdentity();

        P_ *= 0.1;
        Q_ *= 1e-4;
        R_ *= 1e-3;
    }

    void set_initial_state(const Eigen::Vector4d& x0) {
        x_hat_ = x0;
    }

    void set_initial_covariance(const Eigen::Matrix4d& P0) {
        P_ = P0;
    }

    void set_process_noise(const Eigen::Matrix4d& Q) {
        Q_ = Q;
    }

    void set_measurement_noise(const Eigen::Matrix2d& R) {
        R_ = R;
    }

    const Eigen::Vector4d& state() const {
        return x_hat_;
    }

    const Eigen::Matrix4d& covariance() const {
        return P_;
    }

    void predict(double u, double dt, const CartPoleParams& params) {
        Eigen::Matrix4d F = numerical_jacobian(x_hat_, u, dt, params);

        x_hat_ = state_transition(x_hat_, u, dt, params);

        P_ = F * P_ * F.transpose() + Q_;
    }

    void update(const Eigen::Vector2d& y_meas) {
        Eigen::Vector2d y_hat = measurement_model(x_hat_);
        Eigen::Matrix<double, 2, 4> H = measurement_jacobian();

        Eigen::Vector2d innovation = y_meas - y_hat;

        Eigen::Matrix2d S = H * P_ * H.transpose() + R_;
        Eigen::Matrix<double, 4, 2> K = P_ * H.transpose() * S.inverse();

        x_hat_ = x_hat_ + K * innovation;

        Eigen::Matrix4d I = Eigen::Matrix4d::Identity();
        P_ = (I - K * H) * P_;
    }

private:
    Eigen::Vector4d x_hat_;
    Eigen::Matrix4d P_;
    Eigen::Matrix4d Q_;
    Eigen::Matrix2d R_;

    Eigen::Vector4d state_transition(const Eigen::Vector4d& x,
                                     double u,
                                     double dt,
                                     const CartPoleParams& params) const
    {
        return rk4_step(cartpole_dynamics, x, u, dt, params);
    }

    Eigen::Matrix4d numerical_jacobian(const Eigen::Vector4d& x,
                                       double u,
                                       double dt,
                                       const CartPoleParams& params) const
    {
        Eigen::Matrix4d F;
        const double eps = 1e-6;

        Eigen::Vector4d f0 = state_transition(x, u, dt, params);

        for (int i = 0; i < 4; ++i) {
            Eigen::Vector4d x_perturbed = x;
            x_perturbed(i) += eps;

            Eigen::Vector4d fi = state_transition(x_perturbed, u, dt, params);

            F.col(i) = (fi - f0) / eps;
        }

        return F;
    }

    Eigen::Vector2d measurement_model(const Eigen::Vector4d& x) const {
        Eigen::Vector2d y;
        y << x(0), x(2);
        return y;
    }

    Eigen::Matrix<double, 2, 4> measurement_jacobian() const {
        Eigen::Matrix<double, 2, 4> H;
        H.setZero();
        H(0, 0) = 1.0;
        H(1, 2) = 1.0;
        return H;
    }
};

#pragma once
#include <random>
#include <Eigen/Dense>

struct SensorParams {
    double pos_noise_std = 0.01;    // meters
    double angle_noise_std = 0.005; // radians
};

class SensorModel {
public:
    explicit SensorModel(const SensorParams& params)
        : params_(params),
          rng_(std::random_device{}()),
          pos_noise_(0.0, params.pos_noise_std),
          angle_noise_(0.0, params.angle_noise_std)
    {}

    // Returns y = [p_meas, theta_meas]^T
    Eigen::Vector2d measure(const Eigen::Vector4d& x_true) {
        const double p_true = x_true(0);
        const double theta_true = x_true(2);

        Eigen::Vector2d y;
        y << p_true + pos_noise_(rng_),
             theta_true + angle_noise_(rng_);
        return y;
    }

private:
    SensorParams params_;
    std::mt19937 rng_;
    std::normal_distribution<double> pos_noise_;
    std::normal_distribution<double> angle_noise_;
};

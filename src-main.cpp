#include <iostream>
#include <Eigen/Dense>
#include "cartpole.hpp"
#include "rk4.hpp"
#include "csv_logger.hpp"
#include "sensors.hpp"
#include "ekf.hpp"
#include "lqr.hpp"

static double clamp_force(double u, double u_min, double u_max) {
    if (u < u_min) return u_min;
    if (u > u_max) return u_max;
    return u;
}

static double force_input(double t) {
    if (t > 5.0 && t < 5.2) return 5.0;   // push cart right
    if (t > 12.0 && t < 12.2) return -5.0; // push cart left
    return 0.0;
}


int main() {
    try {
        // -----------------------------
        // Plant parameters
        // -----------------------------
        CartPoleParams prm;
        prm.M = 1.0;
        prm.m = 0.1;
        prm.l = 0.5;
        prm.g = 9.81;
        prm.b = 0.05;

        // -----------------------------
        // Simulation timing
        // -----------------------------
        const double dt = 0.02;   // 500 Hz
        const double T  = 1000.0;
        const int steps = static_cast<int>(T / dt);

        // -----------------------------
        // Sensor model
        // -----------------------------
        SensorParams sensor_params;
        sensor_params.pos_noise_std = 0.01;
        sensor_params.angle_noise_std = 0.005;

        SensorModel sensors(sensor_params);

        // -----------------------------
        // EKF setup
        // -----------------------------
        ExtendedKalmanFilter ekf;

        Eigen::Vector4d x0_hat;
        x0_hat << 0.0, 0.0, 0.0, 0.0;
        ekf.set_initial_state(x0_hat);

        Eigen::Matrix4d P0 = Eigen::Matrix4d::Identity();
        P0 *= 0.1;
        ekf.set_initial_covariance(P0);

        Eigen::Matrix4d Q = Eigen::Matrix4d::Identity();
        Q *= 1e-4;
        ekf.set_process_noise(Q);

        Eigen::Matrix2d R = Eigen::Matrix2d::Identity();
        R(0, 0) = sensor_params.pos_noise_std * sensor_params.pos_noise_std;
        R(1, 1) = sensor_params.angle_noise_std * sensor_params.angle_noise_std;
        ekf.set_measurement_noise(R);

        // -----------------------------
        // LQR setup
        // -----------------------------
        LQRController lqr;

        Eigen::Matrix4d A_d;
        Eigen::Matrix<double, 4, 1> B_d;
        get_numerical_discrete_cartpole_linearization(dt, prm, A_d, B_d);

        Eigen::Matrix4d Q_lqr = Eigen::Matrix4d::Zero();
        Q_lqr(0, 0) = 1.0;    // cart position
        Q_lqr(1, 1) = 0.1;     // cart velocity
        Q_lqr(2, 2) = 100.0;   // pendulum angle
        Q_lqr(3, 3) = 10.0;    // pendulum angular velocity

        Eigen::Matrix<double, 1, 1> R_lqr;
        R_lqr(0, 0) = 1.0;     // control effort penalty

        Eigen::Matrix<double, 1, 4> K_lqr =
            compute_discrete_lqr_gain(A_d, B_d, Q_lqr, R_lqr);

        lqr.set_gain(K_lqr);

        // -----------------------------
        // True initial state
        // -----------------------------
        Eigen::Vector4d x;
        x << 0.0, 0.0, 0.3, 0.0;   // 5 deg initial tilt

        // -----------------------------
        // Logging
        // -----------------------------
        CsvLogger logger("sim.csv");
        logger.write_header({
            "t",
            "u",
            "p_true",
            "p_dot_true",
            "theta_true",
            "theta_dot_true",
            "p_meas",
            "theta_meas",
            "p_est",
            "p_dot_est",
            "theta_est",
            "theta_dot_est"
        });

        // -----------------------------
        // Simulation loop
        // -----------------------------
        double t = 0.0;

        for (int k = 0; k < steps; ++k) {
            // Use EKF estimate for feedback
            double u = lqr.control(ekf.state());

            double disturbance = force_input(t);   // external disturbance force
            u += disturbance;

            u = clamp_force(u, -20.0, 20.0);

            // Simulated noisy measurements from true state
            Eigen::Vector2d y_meas = sensors.measure(x);

            // EKF predict/update
            ekf.predict(u, dt, prm);
            ekf.update(y_meas);

            Eigen::Vector4d x_est = ekf.state();

            // Log everything
            logger.write_row(
                t,
                u,
                x(0),        // p_true
                x(1),        // p_dot_true
                x(2),        // theta_true
                x(3),        // theta_dot_true
                y_meas(0),   // p_meas
                y_meas(1),   // theta_meas
                x_est(0),    // p_est
                x_est(1),    // p_dot_est
                x_est(2),    // theta_est
                x_est(3)     // theta_dot_est
            );

            // Advance true nonlinear plant
            x = rk4_step(cartpole_dynamics, x, u, dt, prm);

            // Advance time
            t += dt;
        }

        std::cout << "Done. Wrote sim.csv\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

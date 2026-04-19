#include "cartpole.hpp"
#include <cmath>

Eigen::Vector4d cartpole_dynamics(const Eigen::Vector4d& x,
                                 double u,
                                 const CartPoleParams& prm)
{
    const double pos   = x(0);
    const double vel   = x(1);
    const double th    = x(2);
    const double thdot = x(3);

    (void)pos;

    const double M = prm.M;
    const double m = prm.m;
    const double l = prm.l;
    const double g = prm.g;
    const double b = prm.b;

    const double s = std::sin(th);
    const double c = std::cos(th);

    const double u_eff = u - b * vel;
    const double denom = (M + m) - m * c * c;

    const double p_ddot =
        (u_eff + m * s * (l * thdot * thdot + g * c)) / denom;

    const double th_ddot =
        (-u_eff * c - m * l * thdot * thdot * c * s - (M + m) * g * s)
        / (l * denom);

    Eigen::Vector4d xdot;
    xdot << vel, p_ddot, thdot, th_ddot;
    return xdot;
}

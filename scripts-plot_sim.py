import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sim.csv")

plt.figure()
plt.plot(df["t"], df["p_true"], label="p_true")
plt.plot(df["t"], df["p_meas"], label="p_meas", alpha=0.7)
plt.plot(df["t"], df["p_est"], label="p_est", linestyle="--")
plt.xlabel("time (s)")
plt.ylabel("cart position (m)")
plt.title("Cart Position: True vs Measured vs Estimated")
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(df["t"], df["theta_true"], label="theta_true")
plt.plot(df["t"], df["theta_meas"], label="theta_meas", alpha=0.7)
plt.plot(df["t"], df["theta_est"], label="theta_est", linestyle="--")
plt.xlabel("time (s)")
plt.ylabel("pendulum angle (rad)")
plt.title("Pendulum Angle: True vs Measured vs Estimated")
plt.grid(True)
plt.legend()

plt.figure()
plt.plot(df["t"], df["u"], label="u")
plt.xlabel("time (s)")
plt.ylabel("force (N)")
plt.title("Input Force")
plt.grid(True)
plt.legend()

plt.show()

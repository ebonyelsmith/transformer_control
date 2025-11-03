import numpy as np
from scipy.linalg import solve_continuous_are
from numba import njit
from scipy.signal import place_poles


def batch_lqr_gains(lambda1, lambda2, Q, R):
    """
    # Computes LQR gains for multiple systems in a batch.
    pole placement method is used.
    """
    n = len(lambda1)
    K_all = np.zeros((n, 1, 2))  # shape (systems, 1, 2)
    for i in range(n):
        l1, l2 = lambda1[i], lambda2[i]
        A = np.array([[l1, 0],
                      [0, l2]])
        # A = np.diag([l1, l2])
        
        B = np.array([[1.0],
                      [1.0]])
        # X = solve_continuous_are(A, B, Q, R)
        # K = np.linalg.inv(R) @ (B.T @ X)
        # print(f"A: {A}")
        # print(f"B: {B}")

        desired_poles = np.array([-1, -1.1])
        # desired_poles = np.array([-5, -6])

        # Compute the gain matrix using pole placement
        K = place_poles(A, B, desired_poles).gain_matrix
        # K = np.zeros((1, 2))  # Initialize K as a 1x2 array

        K_all[i] = K
    return K_all  # shape (n, 1, 2)

@njit
# def simulate_batch(X0_list, K_all, dt, n_steps):
def simulate_batch(X0_list, K_all, lambda1, lambda2, dt, n_steps):
    """
    Vectorized RK4 simulation using LQR feedback for multiple systems.
    """
    num_systems = X0_list.shape[0]
    x1_all = np.zeros((num_systems, n_steps))
    x2_all = np.zeros((num_systems, n_steps))
    tau_all = np.zeros((num_systems, n_steps))

    for i in range(num_systems):
        x = X0_list[i]
        K = K_all[i]
        x1_all[i, 0] = x[0]
        x2_all[i, 0] = x[1]
        # mass, length = masses[i], lengths[i]
        l1, l2 = lambda1[i], lambda2[i]

        for t in range(n_steps - 1):
            # if t < 30:
            #     u = np.array([0.0])
            # else:
            u = -K @ (x - np.array([0.0, 0.0]))
            
            # print(f"u: {u}")
            tau_all[i, t] = u[0]
            # x = rk4_step(x, u[0], dt)
            x = rk4_step(x, u[0], dt, l1, l2)
            noise = np.random.normal(0, 1, size=x.shape) * 0.03  # Add noise
            if t > 85:
                noise = np.random.normal(0, 1, size=x.shape) * 0.01  # Add noise
                x += noise
            x1_all[i, t + 1] = x[0]
            x2_all[i, t + 1] = x[1]

    return x1_all, x2_all, tau_all

@njit
def rk4_step(x, u, dt, l1, l2):
    """
    RK4 integrator for a single system step.
    """
    def f(x, u):
        # A = np.diag([l1, l2])
        A = np.array([[l1, 0],
                      [0, l2]])
        B = np.array([[1.0],
                      [1.0]])

        # print(f"A type: {type(A)}")
        # print(f"x type: {type(x)}")
        # print(f"B type: {type(B)}")
        # print(f"u type: {type(u)}")
        
        return A @ x + B.flatten() * u

    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def run_all_simulations(X0_list, total_time, dt, lambda1, lambda2):
    n_steps = int(total_time / dt) + 1
    Q = np.eye(2)
    R = np.array([[1]])
    K_all = batch_lqr_gains(lambda1, lambda2, Q, R)
    x1, x2, tau = simulate_batch(X0_list, K_all, lambda1, lambda2, dt, n_steps) 
    T = np.linspace(0, total_time, n_steps)
    return T, x1, x2, tau


def checking(X0, total_time, lambda1, lambda2, method='rk4', dt=0.05):
    """
    Checks the simulation of inverted pendulum systems using either RK4 or Euler methods.
    
    Args:
        X0 (list or np.ndarray): Initial state of the pendulum [theta, thetadot].
        total_time (float): Total duration of the simulation in seconds.
        lambda1 (float): first eigenvalue for the system.
        lambda2 (float): second eigenvalue for the system.
        method (str, optional): Method to use for simulation ('rk4' or 'euler'). Defaults to 'rk4'.
        dt (float, optional): Time step for the simulation. Defaults to 0.01.
    
    Returns:
        T, theta_all, thetadot_all, tau_all: Simulation results for each system.
    """
    if method == 'rk4':
        # T, theta_all, thetadot_all, tau_all = vectorized_simulation(X0, total_time, dt, masses, lengths)
        # T, theta_all, thetadot_all, tau_all = run_all_simulations(X0, total_time, dt, lambda1, lambda2)
        T, x1_all, x2_all, tau_all = run_all_simulations(X0, total_time, dt, lambda1, lambda2)
    else:
        raise ValueError("Invalid method. Choose 'rk4'.")
    
    # return T, theta_all, thetadot_all, tau_all
    return T, x1_all, x2_all, tau_all


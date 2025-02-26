import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
hbar = 1.0 # Planck's constant (set to 1 in natural units)
m = 1.0    # Particle mass (set to 1 in natural units)

# Spatial grid
x_min = -100.0                   # Minimum position for the spatial domain
x_max = 100.0                    # Maximum position for the spatial domain
dx = 0.1                         # Step size for spatial discretization
x = np.arange(x_min, x_max, dx)  # Position grid array
N = len(x)                       # Number of spatial points

# Momentum grid
dk = 2 * np.pi / (N * dx) # Step size for momentum space (fourier transformation properties)
k = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi

# Time grid
dt = 0.01                    # Time step for evolution
t_max = 10.0                 # Total simulation time
t = np.arange(0, t_max, dt)  # Time grid array
num_t = len(t)               # Number of time steps

# Initial Gaussian wave packet parameters
x0 = -10.0   # Initial position
p0 = 5.0     # Initial momentum
sigma = 1.0  # Width

def psi_initial(x):
    """Initial Gaussian wave packet.
    Combines a Gaussian envelope with a plane wave (complex exponential)."""
    return (1 / (np.sqrt(sigma * np.sqrt(np.pi)))) * np.exp(-((x - x0)**2) / (2 * sigma**2)) * np.exp(1j * p0 * x / hbar)

# Potential barrier parameters
V0 = 1.5  # Height
a = 5.0   # Half-width

def V(x):
    """Defines the potential barrier.
    Returns V0 for positions within [-a, a], and 0 elsewhere."""
    return np.where(np.abs(x) <= a, V0, 0.0)

# Initial wavefunction
psi = psi_initial(x)

# Potential energy operator
V_x = V(x)                                   # Evaluate potential at each spatial point
exp_V = np.exp(-1j * V_x * dt / (2 * hbar))  # Half-step potential operator for time evolution

# Kinetic energy operator in momentum space
T_k = (hbar**2 * k**2) / (2 * m)       # Kinetic energy for a free particle
exp_T = np.exp(-1j * T_k * dt / hbar)  # Full-step kinetic operator

# Prepare for time evolution
psi_t = np.zeros((num_t, N), dtype=complex)
psi_t[0, :] = psi

# Time evolution using the split-operator method
for i in range(1, num_t):
    
    psi = exp_V * psi
    psi_k = np.fft.fftshift(np.fft.fft(psi))
    psi_k = exp_T * psi_k
    psi = np.fft.ifft(np.fft.ifftshift(psi_k))
    psi = exp_V * psi
    norm = np.trapz(np.abs(psi)**2, x) # Normalize the wavefunction to remain total probability at 1.0
    psi /= np.sqrt(norm)
    psi_t[i, :] = psi

def compute_probabilities(psi, x, a):
    """Compute probabilities of reflection, transmission, and barrier region."""
    x_barrier_left = -a
    x_barrier_right = a

    # Reflection region: x < x_barrier_left
    mask_reflect = x < x_barrier_left
    prob_reflect = np.trapz(np.abs(psi[mask_reflect])**2, x[mask_reflect])

    # Transmission region: x > x_barrier_right
    mask_transmit = x > x_barrier_right
    prob_transmit = np.trapz(np.abs(psi[mask_transmit])**2, x[mask_transmit])

    # Probability within the barrier region
    mask_barrier = (x >= x_barrier_left) & (x <= x_barrier_right)
    prob_barrier = np.trapz(np.abs(psi[mask_barrier])**2, x[mask_barrier])

    # Total probability (Sum must be 1.0)
    total_prob = prob_reflect + prob_transmit + prob_barrier

    # Residual probability (negligible due to normalization)
    prob_other = 1.0 - total_prob

    return prob_reflect, prob_transmit, prob_barrier, prob_other

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot(x, np.abs(psi_t[0, :])**2, label='Wave Packet')
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, np.max(np.abs(psi_t[0, :])**2) * 1.2)
ax.set_xlabel('Position x')
ax.set_ylabel(r'$|\Psi(x, t)|^2$')
ax.set_title('Quantum Tunneling Simulation')
ax.legend()

# Plot the potential barrier
ax.plot(x, V_x / np.max(V_x) * np.max(np.abs(psi_t[0, :])**2), label='Potential Barrier', color='red', linestyle='--')

# Dynamic legend for probabilities
prob_reflect, prob_transmit, prob_barrier, prob_other = compute_probabilities(psi_t[0, :], x, a)
legend_text = ax.text(0.05, 0.95, f"Reflect: {prob_reflect:.2f}, Transmit: {prob_transmit:.2f}, Barrier: {prob_barrier:.2f}",
                      transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

# Animate the wave packet evolution
def animate(i):
    """Updates the wavefunction and probabilities for each time step."""
    line.set_ydata(np.abs(psi_t[i, :])**2)
    
    # Recompute probabilities for the current frame
    prob_reflect, prob_transmit, prob_barrier, prob_other = compute_probabilities(psi_t[i, :], x, a)
    legend_text.set_text(f"Reflect: {prob_reflect:.2f}, Transmit: {prob_transmit:.2f}, Barrier: {prob_barrier:.2f}")
    
    return line, legend_text

anim = FuncAnimation(fig, animate, frames=num_t, interval=50, blit=True)

plt.legend(loc='upper right')
plt.show()

# Final probabilities
psi_final = psi_t[-1, :]
prob_reflect, prob_transmit, prob_barrier, prob_other = compute_probabilities(psi_final, x, a)
total_prob = prob_reflect + prob_transmit + prob_barrier + prob_other
print(f"Final Reflection Probability: {prob_reflect:.4f}")
print(f"Final Transmission Probability: {prob_transmit:.4f}")
print(f"Probability in Barrier: {prob_barrier:.4f}")
print(f"Other Probabilities (should be negligible): {prob_other:.4e}")
print(f"Total Probability: {total_prob:.6f}")

""" By Narrendharan Elemaran | 2311510/1 | SQA7018 """
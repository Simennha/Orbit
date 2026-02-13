import numpy as np
import matplotlib.pyplot as plt

"""
Satellite solar-panel pointing / power simulation (constant angular-rate rotation)

This script simulates how a direction vector (typically a solar panel normal or
any body-fixed “pointing” vector) evolves over time when the satellite rotates
with a constant angular velocity vector ω = [ωx, ωy, ωz]. It then computes the
instantaneous solar power produced based on the cosine of the angle between the
rotating normal vector and a fixed Sun direction, with negative cosine values
clipped to 0 (no illumination) and with an optional maximum power cap.

Coordinate frame / interpretation
---------------------------------
All vectors in this script (v0, omega_xyz, sun_vector) are expressed in the SAME
Cartesian coordinate system with fixed axes:
    x-axis = (1, 0, 0)
    y-axis = (0, 1, 0)
    z-axis = (0, 0, 1)

Important: the Sun vector is NOT automatically an axis. It is just a direction
vector given in the chosen coordinate frame. If you set sun_vector=(1,0,0),
then the Sun happens to align with +x; otherwise it points in a general
direction.

Rotation model
--------------
The code assumes a constant angular velocity vector ω. Over each timestep dt,
the vector v is rotated by an angle:
    dθ = |ω| * dt
around the axis:
    k = ω / |ω|

The 3×3 rotation matrix for this incremental rotation is computed using
Rodrigues’ rotation formula:
    R = I + sin(dθ) K + (1 - cos(dθ)) K^2
where K is the skew-symmetric cross-product matrix of k.

The update step is:
    v_{n+1} = R * v_n

This corresponds to rotation about a fixed axis in the chosen reference frame.
(If you need full rigid-body attitude dynamics or torque-driven motion, you
would propagate an attitude matrix or quaternion instead.)

Illumination / cosine model
---------------------------
At each timestep, the script computes:
    cos(theta) = (v · s) / (|v| |s|)
where s is the Sun direction. Negative values are clipped to 0, representing
times when the panel normal points more than 90° away from the Sun (no direct
illumination):
    cos(theta) = max(cos(theta), 0)

Power model
-----------
A simple power model is used:
    power = min(cos(theta) * innstrålet_over_areal, max_power)

Where:
- innstrålet_over_areal is a constant that can represent:
    solar_irradiance * panel_area * efficiency
  (units depend on your chosen constants; keep them consistent)
- max_power caps the output at a maximum generation level.

Main functions
--------------
1) rodrigues_rotation_matrix(axis, angle)
   Returns a 3×3 rotation matrix that rotates vectors by 'angle' radians about
   the given 'axis' vector.

2) simulate_cos_angle(v0, omega_xyz, t_end, dt, sun_vector)
   Simulates the rotation of v starting from v0, using constant ω, for the time
   interval [0, t_end] with timestep dt.
   Returns:
     - t: time array
     - power_vals: instantaneous power at each time
     - mean_val: mean of the *clipped cosine* over the simulation (note: if you
       want to optimize mean power instead, replace mean_val with mean(power_vals))

3) optimal_rotation_vector(v0, do, search_range, t_end, dt, sun_vector)
   Brute-force searches over a grid of candidate ω components and finds the
   ω that maximizes the objective returned by simulate_cos_angle (currently the
   mean clipped cosine).
   The search grid is:
       ωx = 0.1*i, ωy = 0.1*j, ωz = 0.1*k  for i,j,k in search_range
   Returns:
     - best_vals: power time series for the best ω
     - best_vec: the best ω vector found
     - best_mean: the best objective value found

How to use
----------
1) Set the constants:
   - max_power
   - innstrålet_over_areal
   - time (t_end) and d_time (dt)

2) Choose initial conditions:
   - v0: initial pointing/normal vector (non-zero)
   - omega_xyz: angular velocity vector [ωx, ωy, ωz]
   - optionally change sun_vector inside function calls

3) Run a simulation:
   t, power_vals, mean_val = simulate_cos_angle(v0, omega_xyz)

4) Optionally find a “best” ω via brute-force search:
   best_power_vals, optimal_rot_vec, best_mean = optimal_rotation_vector(v0, do=True)

Notes / common pitfalls
-----------------------
- dt controls both accuracy and runtime. Smaller dt -> more steps -> slower.
- The brute-force optimizer can be expensive: it runs simulate_cos_angle many times.
  Start with small search ranges (e.g., range(-5,6)) and increase if needed.
- Make sure the dt and t_end used in the optimizer match your intended simulation
  if you want arrays to be comparable/plot together.
- If you want the optimizer to maximize average power instead of average cosine,
  change:
      mean_val = np.mean(cos_vals)
  to:
      mean_val = np.mean(power_vals)

"""

max_power = 3*7270                    # Max power generation
innstrålet_over_areal = 1361000*0.3*0.3*0.295 #Stråling*areal*effektivitet
time = 60 #Simulasjonstid
d_time = 0.2 #time steps
sun_vector = (1.0, 0.0, 0.0)


def rodrigues_rotation_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0.0:
        return np.eye(3)

    k = axis / axis_norm
    kx, ky, kz = k

    K = np.array([
        [0,  -kz,  ky],
        [kz,  0,  -kx],
        [-ky, kx,  0]
    ], dtype=float)

    I = np.eye(3)
    return I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def simulate_cos_angle(v0, omega_xyz, t_end = time, dt=d_time, sun_vector=(1.0, 0.0, 0.0)):
    v = np.asarray(v0, dtype=float)
    omega = np.asarray(omega_xyz, dtype=float)
    sun = np.asarray(sun_vector, dtype=float)

    if np.linalg.norm(v) == 0:
        raise ValueError("Initial vector v0 must be non-zero.")
    if np.linalg.norm(sun) == 0:
        raise ValueError("Sun vector must be non-zero.")

    n_steps = int(np.floor(t_end / dt)) + 1
    t = np.linspace(0.0, dt * (n_steps - 1), n_steps)
    cos_vals = np.empty(n_steps, dtype=float)

    omega_mag = np.linalg.norm(omega)
    sun_norm = np.linalg.norm(sun)

    if omega_mag == 0.0:
        cos_const = np.dot(v, sun) / (np.linalg.norm(v) * sun_norm)
        cos_vals.fill(cos_const)
    else:
        axis = omega / omega_mag
        dtheta = omega_mag * dt
        Rdt = rodrigues_rotation_matrix(axis, dtheta)

        v_norm = np.linalg.norm(v)
        denom = v_norm * sun_norm
        for i in range(n_steps):
            cos_vals[i] = np.dot(v, sun) / denom
            v = Rdt @ v

    cos_vals[cos_vals < 0] = 0.0
    
    power_vals = np.minimum(cos_vals*innstrålet_over_areal, max_power)

    mean_val = np.mean(cos_vals)

    return t, power_vals, mean_val


def optimal_rotation_vector(v0, do = True, search_range=range(-10, 11), t_end = time, dt= d_time, sun_vector=(1.0, 0.0, 0.0)):
    if do == False: 
        return None, 0
    
    best_mean = -np.inf
    best_vec = None

    for i in search_range:
        for j in search_range:
            for k in search_range:
                _, power_vals, mean_cos = simulate_cos_angle(
                    v0, [i*0.1, j*0.1, k*0.1],
                    t_end=t_end, dt = dt,
                    sun_vector = sun_vector
                )
                if mean_cos > best_mean:
                    best_vals = power_vals
                    best_mean = mean_cos
                    best_vec = [i*0.1, j*0.1, k*0.1]
    return best_vals, best_vec, best_mean


# inputs
v0 = [0, 2, 5]                 # initial vector
omega_xyz = [0.1, 0.1, 0.1]      # angular velocity

find_best_rot_vec = True           


t, power_vals, mean_cos = simulate_cos_angle(
    v0, omega_xyz
)

best_power_vals, optimal_rot_vec, best_mean = optimal_rotation_vector(v0, do = find_best_rot_vec)



print("Gjennomsnittlig cos-verdi når negative cos-verdier er satt til 0:", mean_cos)
print(f'Power generation: {round(np.mean(power_vals), 3)} mW')
print(f'The optimal vector for v0 is {optimal_rot_vec}, which gives the mean value {round(best_mean, 3)}')

plt.figure()
plt.plot(t, power_vals)
plt.plot(t, best_power_vals)
plt.xlabel("Time (s)")
plt.ylabel("cos(theta(normal vector(t), sun_vector))")
plt.title("Simulated power generation")
plt.grid(True)
plt.show()
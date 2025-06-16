import mujoco
import numpy as np

# Load your model
model = mujoco.MjModel.from_xml_path("models/single_catch.xml")
data = mujoco.MjData(model)

# Update positions
mujoco.mj_fwdPosition(model, data)

ball_id = model.body('ball').id

print(data.xpos[ball_id])


origin = np.array(data.xpos[ball_id], dtype=np.float64)

starting_angle = 0  # Starting angle in degrees
fov = 45 # degrees
nray = 21 # prefer odd number for symmetry
angle_step = fov / (nray - 1)  # Angle step between rays

angles = np.linspace(starting_angle - fov / 2, starting_angle + fov / 2, nray)  # Generate angles for rays

x = np.cos(np.radians(angles))  # Sine for x direction
z = np.sin(np.radians(angles))  # Sine for x direction
directions = np.column_stack((x, z))  # Cosine for z direction
directions = directions.flatten().astype(np.float64)  # Flatten to 1D array


# Define origin of the ray
# origin = np.array([0.0, 0.0, 1.0], dtype=np.float64)

# Define ray directions (flattened array of nray*3)
# Let's say you want 2 rays
# directions = np.array([
#     0.0, 0.0, -1.0,   # Ray 1: pointing straight down
#     1.0, 0.0, -1.0    # Ray 2: diagonally down
# ], dtype=np.float64)

# # Number of rays

# # Output arrays
# geomid = np.zeros(nray, dtype=np.int32)
# dist = np.zeros(nray, dtype=np.float64)

# # Optional: filter groups
# geomgroup = None  # or np.array([1]*model.ngeom, dtype=np.uint8)

# # Call raycast
# mujoco.mj_multiRay(model, data, origin, directions, 
#                    geomgroup, 
#                    flg_static=1,            # Include static geoms
#                    bodyexclude=-1,          # No exclusion
#                    geomid=geomid,
#                    dist=dist,
#                    nray=nray,
#                    cutoff=10.0)             # Max ray length

# # Output
# for i in range(nray):
#     if dist[i] < 0:
#         print(f"Ray {i}: No hit")
#     else:
#         print(f"Ray {i}: Hit geom {geomid[i]} at distance {dist[i]}")
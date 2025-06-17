import mujoco
import numpy as np

# Load your model
model = mujoco.MjModel.from_xml_path("models/single_catch.xml")
data = mujoco.MjData(model)

# Update positions
mujoco.mj_fwdPosition(model, data)

ball_id = model.body('ball').id
cam_id = model.camera('eye0').id

origin = np.array(data.xpos[cam_id], dtype=np.float64)

starting_angle = -45  # Starting angle in degrees
fov = 45 # degrees
nray = 221 # number of rays
angle_step = fov / (nray - 1)  # Angle step between rays

angles = np.linspace(starting_angle - fov / 2, starting_angle + fov / 2, nray)  # Generate angles for rays\

x = np.cos(np.radians(angles))  # Sine for x direction
y = np.zeros(nray)
z = np.sin(np.radians(angles))  # Sine for x direction
directions = np.column_stack((x, y, z))  # Cosine for z direction
directions = directions.flatten().astype(np.float64)  # Flatten to 1D array

# Output arrays
geomid = np.zeros(nray, dtype=np.int32)
dist = np.zeros(nray, dtype=np.float64)

# Optional: filter groups
# geomgroup = None  # or np.array([1]*model.ngeom, dtype=np.uint8)
# geomgroup = np.array([0]*model.ngeom, dtype=np.uint8)
# for i in range(model.ngeom):
#     if model.geom_group[i] == 3:
#         geomgroup[i] = 1
geomgroup = np.array([0,0,0,1,0,0], dtype=np.uint8)

print(origin.shape)
print(directions.shape)
print(geomgroup.shape)

# Call raycast
mujoco.mj_multiRay(model, data, origin, directions, 
                   geomgroup, 
                   flg_static=1,            # Include static geoms
                   bodyexclude=-1,          # No exclusion
                   geomid=geomid,
                   dist=dist,
                   nray=nray,
                   cutoff=10.0)             # Max ray length

# for i in range(nray):
#     start = origin[i]
#     if geomid[i] != -1:
#         # Hit something — draw to hit point
#         end = origin[i] + dist[i] * directions[i]
#         rgba = [0, 1, 0, 1]  # green
#     else:
#         # No hit — draw to cutoff distance
#         end = origin[i] + 10.0 * directions[i]
#         rgba = [1, 0, 0, 1]  # red

#     mjv_addLine(scene, start, end, rgba)

# Output
for i in range(nray):
    if dist[i] < 0:
        print(f"Ray {i}: No hit")
    else:
        print(f"Ray {i}: Hit geom {geomid[i]} at distance {dist[i]}")

ball_geom_id = model.body('ball').id
ball_pos = data.geom_xpos[ball_geom_id]  # shape (3,)

print(ball_pos)
print(data.xpos[cam_id])

import matplotlib.pyplot as plt

# Assumes you have these already:
# - origin: (nray, 3)
# - directions: (nray, 3)
# - dist: (nray,)
# - geomid: (nray,)
# - ball_geom_id: ID of the ball geom
# - model, data

# Get ball position (center of its geom)

# Plot setup
plt.figure(figsize=(8, 6))

start = origin
for i in range(nray):
    if geomid[i] != -1:
        end = start + dist[i] * directions[i]
        color = 'green' if geomid[i] == ball_geom_id else 'gray'
    else:
        end = start + 10.0 * directions[i]
        color = 'red'

    # Plot the ray line in X-Z
    plt.plot(start[0], start[2], end[0], end[2], color=color, alpha=0.6)

# Mark ray origins
plt.scatter(origin[0], origin[2], c='blue', marker='o', s=100, edgecolors='black', label='Ray Origins')

# Mark the ball position
plt.scatter(ball_pos[0], ball_pos[2], c='orange', marker='o', s=100, edgecolors='black', label='Ball Position')

# Labels and legend
plt.xlabel("X")
plt.ylabel("Z")
plt.title("Raycast Visualization in X-Z Plane")
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
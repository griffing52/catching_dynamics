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

starting_angle = -45  # Starting angle in degrees
fov = 45 # degrees
nray = 3 # number of rays
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
geomgroup = np.array([0]*model.ngeom, dtype=np.uint8)
for i in range(model.ngeom):
    if model.geom_group[i] == 3:
        geomgroup[i] = 1

# Call raycast
mujoco.mj_multiRay(model, data, origin, directions, 
                   geomgroup, 
                   flg_static=1,            # Include static geoms
                   bodyexclude=-1,          # No exclusion
                   geomid=geomid,
                   dist=dist,
                   nray=nray,
                   cutoff=10.0)             # Max ray length

# Output
for i in range(nray):
    if dist[i] < 0:
        print(f"Ray {i}: No hit")
    else:
        print(f"Ray {i}: Hit geom {geomid[i]} at distance {dist[i]}")
import mujoco
import mujoco.viewer
import time
import os

# Try to load a sample model included with the package or provide a path to your own
# This path might vary slightly depending on your Python environment structure.
# A common location for built-in models after `pip install mujoco` is within the site-packages.
# For simplicity, first download an example XML like humanoid.xml from the MuJoCo GitHub repository
# or use a very simple inline model.

# Simple inline XML model (a box and a sphere)
xml_model = """
<mujoco>
  <worldbody>
    <body>
      <joint type="free"/>
      <geom type="box" size=".1 .2 .3" rgba="1 0 0 1"/>
      <site name="tip" pos="0 0 .3"/>
    </body>
    <body pos="0.5 0 0">
      <joint type="free"/>
      <geom type="sphere" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
  <actuator>
    </actuator>
</mujoco>
"""

# If you have humanoid.xml, you can use:
# model_path = 'dog.xml' # Replace with the actual path if you download one
model_path = 'models/manipulator.xml' # Replace with the actual path if you download one
# model_path = 'arm26.xml' # Replace with the actual path if you download one
# model_path = 'humanoid.xml' # Replace with the actual path if you download one
try:
    model = mujoco.MjModel.from_xml_path(model_path)
except Exception as e:
    print(f"Error loading XML from path: {e}")
    print("Falling back to inline model.")
    model = mujoco.MjModel.from_xml_string(xml_model)

# model = mujoco.MjModel.from_xml_string(xml_model)
data = mujoco.MjData(model)

# Use the new passive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
  # Close the viewer automatically after 60 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 300:
    step_start = time.time()
 
    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(model, data)

    # Synchronize the viewer
    viewer.sync()

    # Rudimentary time keeping, adjust as needed.
    time_until_next_step = model.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
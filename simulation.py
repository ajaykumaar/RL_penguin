import mujoco as mj
import numpy as np
import cv2

model = mj.MjModel.from_xml_path("test_robot.xml")
data = mj.MjData(model)
renderer = mj.Renderer(model,480,640)


mj.mj_forward(model, data)
renderer.update_scene(data,"track")
#print(renderer.render())
freq=1.5

angular_freq=2*np.pi*freq
amplitude=0.011#np.pi/100

fps=60
duration=10
#print(data.ne)
cnt=0
frames = []
print(data.qpos)
#print(data.qvel)
t = 0
#Q usefull: [0-8,11-12,17]
while data.time<duration:
    ctrl = amplitude*np.sin(angular_freq*data.time)
    #data.ctrl=np.array([ctrl,ctrl,-ctrl*4/4]) #uncomment this line for test_robot.xml 
    data.ctrl=np.array([ctrl,ctrl])
    mj.mj_step(model, data)
    #mj.mj_forward(model,data)
    if data.qpos[2] < .09:
        print('robot has fallen')
        
        time_old = data.time
        mj.mj_resetData(model,data)
        #print(data.qpos)
        #print(data.qvel)
        data.time = time_old
    if len(frames) < data.time * fps:
        renderer.update_scene(data,"track")
        frame = renderer.render().copy()
        frames.append(frame)
        #print(data.cinert)
        #print(data.cvel)
cv2.destroyAllWindows()

#print(model.cam_bodyid)
# cv2.destroyAllWindows()
wait_time = int(1/fps * 1000)
for f in frames:
    cv2.imshow('frame',f)
    cv2.waitKey(wait_time)
cv2.destroyAllWindows()
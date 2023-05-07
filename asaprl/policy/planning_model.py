import matplotlib.pyplot as plt
import math, pdb
import numpy as np
import matplotlib.pyplot as plt
import time


class PathParam():
    def __init__(self, lat0, yaw0, lon1, lat1, yaw1, lon_final=80):
        '''
            cubic polunomial curve: lat = a0 + a1 * lon + a2 * lon^2 + a3 * lon^3
            augument:
                lat0: current lateral position
                yaw0: current yaw angle, in degree
                lon1: ending longitudinal position
                lat1: ending lateral position
                yaw1: ending yaw angle, in degree
                lon_final: the longitudinal distance horizon
            return:
                lon: lateral position (with precision of 0.1m)
                lat: lateral position (corresponding to lon)
        '''
        yaw0 = math.tan(yaw0/180*math.pi)
        yaw1 = math.tan(yaw1/180*math.pi)
        self.Horizon = lon1 
        self.lon_final = lon_final 
        self.a0 = lat0
        self.a1 = yaw0
        self.a3 = (2*yaw0 + self.a1*self.Horizon + yaw1*self.Horizon - 2*lat1) / (self.Horizon**3)
        self.a2 = (yaw1 - self.a1 - 3 * self.a3 * (self.Horizon**2) ) / (2*self.Horizon)

        self.GetPathProfile()


    def GetPathProfile(self):
        cal_length = lambda pos:(np.sum(np.sqrt(np.sum(np.square((pos[1:,:2] - pos[:-1,:2])),axis=1)),axis=0))

        self.lon = np.arange(self.Horizon*10) / 10
        self.lat = self.a0 + self.a1 * self.lon + self.a2 * (self.lon**2) + self.a3 * (self.lon**3)

        self.lon = np.expand_dims(self.lon,1)
        self.lat = np.expand_dims(self.lat,1)

        self.yaw = np.arctan((self.lat[1:] - self.lat[:-1]) / (self.lon[1:] - self.lon[:-1])) / math.pi * 180
        self.yaw = np.vstack((self.yaw, self.yaw[-1]))

        self.path = np.hstack((self.lon, self.lat, self.yaw))
        self.path_length = [cal_length(self.path[:i+1,:2]) for i in range(len(self.path-1))]


    def GetPosFromLength(self, s):
        cal_length = lambda pos:(np.sum(np.sqrt(np.sum(np.square((pos[1:,:2] - pos[:-1,:2])),axis=1)),axis=0))

        if self.path_length[-1] < s:
            return None
        matched_ind = min(np.where(np.array(self.path_length)-s >= 0)[0])
        if matched_ind == 0:
            return self.path[:1,:]
        else:
            pos = np.zeros((1,3))
            percent = (s - cal_length(self.path[:matched_ind, :])) / cal_length(self.path[matched_ind-1:matched_ind+1, :])
            pos[0,0] = self.path[matched_ind-1, 0] + percent * (self.path[matched_ind, 0] - self.path[matched_ind-1, 0])
            pos[0,1] = self.path[matched_ind-1, 1] + percent * (self.path[matched_ind, 1] - self.path[matched_ind-1, 1])
            pos[0,2] = self.path[matched_ind-1, 2] + percent * (self.path[matched_ind, 2] - self.path[matched_ind-1, 2])
            return pos

    def PlotPath(self, ax=None):
        if ax:
            ax.plot(self.path[:,0], self.path[:,1])
        else:
            plt.plot(self.path[:,0], self.path[:,1])

class SpeedParam():
    def __init__(self, v0=0, acc0=0, v1=0, acc1=0, stop_time=1, speed_pattern='forward2', T=3):
        '''
            cubic polynomial speed profile: v = a0 + a1 * t + a2 * t^2 + a3 * t^3
            augument:
                v0: current speed
                acc0: current acceleration
                v1: at 'forward' pattern, the ending speed
                stop_time: at 'brake' pattern, the brake time
                speed_pattern: pattern of speed profile, 'brake' or 'forward'
                T: time horizon
            return:
                self.t: time steps
                self.s: speed at each time steps
            patten:
                forward1: with constraint - v0, v1, a0, a1=0
                forward2: with constraint - v0, v1, a0, a1
                stop: with constraint - v0, v1, a0, a1
        '''
        self.T = T
        self.stop_time = stop_time
        self.speed_pattern = speed_pattern

        if self.speed_pattern == 'forward1':
            self.a0 = v0
            self.a1 = acc0
            self.a3 = (self.a1 * T + 2 * self.a0 - 2 * v1) / (T**3)
            self.a2 = (-self.a1 - 3 * self.a3 * (T**2)) / (2 * T)
        elif self.speed_pattern == 'forward2':
            self.a0 = v0
            self.a1 = acc0
            self.a3 = (self.a1 * T + 2 * self.a0 - 2 * v1) / (T**3)
            self.a2 = (acc1 - self.a1 - 3 * self.a3 * (T**2)) / (2 * T)
        else:
            self.a0 = v0
            self.a1 = acc0
            self.a3 = (2 * self.a0 + self.a1 * stop_time) / (stop_time ** 3)
            self.a2 = (-self.a1 - 3 * self.a3 * stop_time**2) / (2 * stop_time)

        self.pts_10 = []    # time steps where the speed crosses 10 m/s
        self.pts_0 = []     # time steps where the speed crosses 0 m/s

        self.GetSpeedProfile()

    def GetSpeedProfile(self):
        if 'forward' in self.speed_pattern:
            self.t = np.arange((self.T+0.1)*10) / 10
            self.s = self.a0 + self.a1 * self.t + self.a2 * (self.t**2) + self.a3 * (self.t**3)
        else:
            t = np.arange((self.stop_time+0.1)*10) / 10
            s = self.a0 + self.a1 * t + self.a2 * (t**2) + self.a3 * (t**3)
            
            self.t = np.arange(int((self.T+0.1)*10)) / 10
            self.s = np.array([s[i] if i < len(s) else 0 for i in range(int((self.T+0.1)*10)) ])

        return self.t, self.s

    def GetSpeed(self, t):
        if t > self.T:
            raise Exception ('GetSpeed method: The specified time exceeds the time horizon of speed profile')

        if 'forward' in self.speed_pattern:
            return self.a0 + self.a1 * t + self.a2 * (t**2) + self.a3 * (t**3)
        else:
            if t <= self.stop_time:
                return self.a0 + self.a1 * t + self.a2 * (t**2) + self.a3 * (t**3)
            else:
                return self.a0 + self.a1 * self.stop_time + self.a2 * (self.stop_time**2) + self.a3 * (self.stop_time**3)

    def GetDistance(self, t):
        if t > self.T + 0.01:
            raise Exception ('GetDistance method: The specified time exceeds the time horizon of speed profile')

        if 'forward' in self.speed_pattern:
            return 1.0 / 4.0 * self.a3 * pow(t,4) + 1.0 / 3.0 * self.a2 * pow(t,3) + 1.0 / 2.0 * self.a1 * pow(t,2) + self.a0 * t
        else:
            if t <= self.stop_time:
                return 1.0 / 4.0 * self.a3 * pow(t,4) + 1.0 / 3.0 * self.a2 * pow(t,3) + 1.0 / 2.0 * self.a1 * pow(t,2) + self.a0 * t
            else:
                return 1.0 / 4.0 * self.a3 * pow(self.stop_time,4) + 1.0 / 3.0 * self.a2 * pow(self.stop_time,3) + 1.0 / 2.0 * self.a1 * pow(self.stop_time,2) + self.a0 * self.stop_time

    def PlotSpeed(self, ax=None):
        if ax:
            ax.plot(self.t, self.s)
        else:
            plt.plot(self.t, self.s)

def dynamic_constraint(lat1, yaw1, current_v, current_a, v1, T=3, lon1=30):
    # turning constraint
    min_turning_radius = 8
    if lon1 <= min_turning_radius: # 
        lat1 = np.clip(lat1, -lon1, lon1)

    # acc constraint
    max_acc = 2
    v1 = np.clip(v1, current_v - T * max_acc, current_v + T * max_acc)

    return lat1, yaw1, current_v, current_a, v1

def dist_constraint(dist_lst):
    max_diff = 1
    negative_value = 0

    negative_info = {}
    exceed_info = {}
    dist_array = np.array(dist_lst)
    dist_diff = dist_array[1:] - dist_array[:-1]

    for i in np.where(dist_diff < negative_value)[0]:
        negative_info[i+1] = dist_diff[i]
    for i in np.where(dist_diff > max_diff)[0]:
        exceed_info[i+1] = dist_diff[i] - max_diff

    for step, value in negative_info.items():
        dist_array[step:] -= value
    for step, value in exceed_info.items():
        dist_array[step:] -= value

    return dist_array.tolist()

def motion_skill_model(lat1, yaw1, current_v, current_a, v1, horizon):
    lat1, yaw1, current_v, current_a, v1 = dynamic_constraint(lat1, yaw1, current_v, current_a, v1, horizon)
    #print("lon1: ", lon1, "lat1: ", lat1, "yaw1: ", yaw1, "   v1: ", v1, "    horizon： ", horizon, "   current_v: ", current_v)
    Path = PathParam(lat0=0, yaw0=0, lon1=30, lat1=lat1, yaw1=yaw1)
    SpeedProfile = SpeedParam(v0=current_v, acc0=current_a, v1=v1)

    # calculate distance travelled via speed profile
    dist_lst = []
    for t in [0.1 * step for step in range(horizon + 1)]:
        dist_lst.append(SpeedProfile.GetDistance(t))
    dist_lst = dist_constraint(dist_lst)

    # project distance onto path to get traj
    traj = np.zeros((horizon + 1, 4))
    for dist_num, s in enumerate(dist_lst):
        pos = Path.GetPosFromLength(s)
        traj[dist_num, :3] = pos

    traj[:-1,3] = np.sqrt(np.sum(np.square(traj[1:, :2] - traj[:-1, :2]),axis=1)) * 10
    traj[-1,3] = traj[-2,3]
    traj[:, [2, 3]] = traj[:, [3, 2]]

    return traj, lat1, yaw1, v1

''' example to use the parameterized motion planning model '''
# lat1 = 8
# yaw1 = 30
# v1 = 8

# current_v = 5
# current_a = 0
# acc1 = 1
# horizon = 10

# lon1 = 30

# motion_skill_model(lat1, yaw1, current_v, current_a, v1, horizon)

''' plot lat '''
# ax = plt.gca()
# for lat1 in range(-30, 30, 2):
#     for yaw1 in range(-30, 30, 5):
#         path = PathParam(lat0=0, yaw0=0, lon1=lon1, lat1=lat1, yaw1=yaw1)
#         path.PlotPath(ax)
#     ax.set_aspect('equal')
#     plt.hlines(0.8, 0, 3.5)
#     plt.hlines(-0.8, 0, 3.5)
#     plt.savefig("figure/lat1_{}.png".format(lat1))
#     plt.close()
#     ax = plt.gca()

''' plot yaw '''
# ax = plt.gca()
# for yaw1 in range(-30, 30, 5):
#     for lat1 in range(-30, 30, 2):
#         path = PathParam(lat0=0, yaw0=0, lon1=lon1, lat1=lat1, yaw1=yaw1)
#         path.PlotPath(ax)
#     ax.set_aspect('equal')
#     ax.hlines(0.8, 0, 3.5)
#     ax.hlines(-0.8, 0, 3.5)
#     plt.savefig("figure/yaw1_{}.png".format(yaw1))
#     plt.close()
#     ax = plt.gca()

import math
import numpy as np
#from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
import quaternion


class sensor_fusion(object):
    def __init__(self, ori, time_T):
        self.roll_a = 0.0
        self.pitch_a = 0.0
        self.yaw_a = 0.0
        self.time_T = time_T
        self.DT = 0.0
        self.Orientation_acc = ori

    def set_angles(self, acc, mag, time_T):

        """
        # ----------------------------------
        self.R_U[1][1] = math.cos(self.roll)
        self.R_U[1][2] = -math.sin(self.roll)
        self.R_U[2][1] = math.sin(self.roll)
        self.R_U[2][2] = math.cos(self.roll)
        #----------------------------------
        self.R_V[0][0] = math.cos(self.pitch)
        self.R_V[0][2] = math.sin(self.pitch)
        self.R_V[2][0] = -math.sin(self.pitch)
        self.R_V[2][2] = math.cos(self.pitch)
        # ----------------------------------
        self.R_W[0][0] = math.cos(self.yaw)
        self.R_W[0][1] = -math.sin(self.yaw)
        self.R_W[1][0] = math.sin(self.yaw)
        self.R_W[1][1] = math.cos(self.yaw)
        # ----------------------------------
        self.Rotation = np.matmul(np.matmul(self.R_W,self.R_V),self.R_U)
        self.Orientation = np.matmul(self.Rotation,self.Orientation)
        self.gravity =  np.matmul(np.linalg.inv(self.Orientation),np.array([0,0,9.8]).transpose())
        """
        # ------------------------------------
        self.DT = time_T
        acc = np.array([acc[0], acc[1], acc[2]]).transpose()
        self.Orientation_acc[:, 2] = acc
        self.Orientation_acc[:, 1] = np.cross(acc, np.array([mag[0], mag[1], mag[2]]).transpose())
        self.Orientation_acc[:, 0] = np.cross(self.Orientation_acc[:, 1], acc)
        self.Orientation_acc[:, 0] = self.Orientation_acc[:, 0] / np.linalg.norm(self.Orientation_acc[:, 0])
        self.Orientation_acc[:, 1] = self.Orientation_acc[:, 1] / np.linalg.norm(self.Orientation_acc[:, 1])
        self.Orientation_acc[:, 2] = self.Orientation_acc[:, 2] / np.linalg.norm(self.Orientation_acc[:, 2])

        tmp = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], float)
        mat = np.matmul(tmp,self.Orientation_acc)
        quat = quaternion.from_rotation_matrix(self.Orientation_acc)
        q = quaternion.as_float_array(quat)
        # --------------------------------------
        self.yaw_a = (math.atan2(2.0 * (q[1] * q[2] - q[0] * q[3]),
                                 -1 + 2 * (q[0] * q[0] + q[1] * q[1])))
        pitch_a = (-math.asin(2.0 * (q[1] * q[3] + q[0] * q[2])))
        roll_a = (math.atan2(2.0 * (-q[0] * q[1] + q[2] * q[3]),
                             -1 + 2 * (q[0] * q[0] + q[1] * q[1])))
        #print(self.yaw_a)
        # -----------------------------------------
        """yaw_g =  (math.atan2(2.0 * (q_gy[1] *q_gy[2] - q_gy[0] * q_gy[3]),
                                                        -1+2*(q_gy[0] * q_gy[0] + q_gy[1] * q_gy[1])))
        pitch_g = (-math.asin(2.0 * (q_gy[1] * q_gy[3] + q_gy[0] * q_gy[2])))
        roll_g = (math.atan2(2.0 * (-q_gy[0] * q_gy[1] + q_gy[2] * q_gy[3]),
                                  -1+2*(q_gy[0] * q_gy[0] + q_gy[1] * q_gy[1])))
        #print(math.degrees(roll_a),math.degrees(pitch_a),math.degrees(yaw_a),math.degrees(roll_g),math.degrees(pitch_g),math.degrees(yaw_g))
        #-----------------------------------------
        #q_final = 0.8*self.q+(1-0.8)*self.quat_gy
        #self.q_final = q_final
        #-----------------------------------------

        yaw =  (math.atan2(2.0 * (q_final[1] *q_final[2] - q_final[0] * q_final[3]),
                                                        -1+2*(q_final[0] * q_final[0] + q_final[1] * q_final[1])))
        pitch = (-math.asin(2.0 * (q_final[1] * q_final[3] + q_final[0] * q_final[2])))
        roll = (math.atan2(2.0 * (-q_final[0] * q_final[1] + q_final[2] * q_final[3]),
                                  -1+2*(q_final[0] * q_final[0] + q_final[1] * q_final[1])))
        #print(math.degrees(roll),math.degrees(pitch),math.degrees(yaw))
        plt.pause(0.001)"""






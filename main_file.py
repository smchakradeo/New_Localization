import socket
import math
import random
import json
import numpy as np
import time
import matplotlib.pyplot as plt
from new_own_trial import sensor_fusion
import quaternion
from future import *

class main_Class(object):
    def __init__(self):
        self.Phi = np.identity(15, float)
        self.B = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], float)
        self.H = np.zeros((10, 9), float)   #--
        self.Pk = np.diag([0, 0, 0, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0, 0, 0, 0.01, 0.01, 0.01])
        arr = np.array([0.0001, 0.0001, 0.0001, 0, 0, 0, 0, 0, 0, 0.0008,0.0008,0.0008,0,0,0],float)
        self.Qk = np.diag(arr)
        self.calib_result = np.array([0, 0, 0], float)
        self.ini_ori = np.identity(3)
        self.orientation = self.ini_ori
        self.x_states = np.array([0, 0, 0, 0, 0, 0], float)
        self.error_states = np.array([0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0, 0.0,0.0,0.0]).reshape(-1,1)
        self.U = np.array([[0, 0, 0, 0, 0, 0]], float).T
        self.time_t = 0.0
        self.DT = 0.0
        self.quat_gy = quaternion.quaternion(0,0,0,0)
        self.z = np.zeros([1, 10], float)


    def first_init(self,data,t):
        mag = np.array([data[3], data[4], data[5]]).transpose()
        grav = np.array([data[0], data[1], data[2]]).transpose()
        self.ini_ori[:, 2] = grav
        self.ini_ori[:, 1] = np.cross(grav, mag)
        self.ini_ori[:, 0] = np.cross(self.ini_ori[:, 1], grav)
        self.ini_ori[:, 2] = self.ini_ori[:, 2] / np.linalg.norm(self.ini_ori[:, 2])
        self.ini_ori[:, 1] = self.ini_ori[:, 1] / np.linalg.norm(self.ini_ori[:, 1])
        self.ini_ori[:, 0] = self.ini_ori[:, 0] / np.linalg.norm(self.ini_ori[:, 0])
        self.time_t = float(t)
        tmp = np.array([[1,0,0],[0,1,0],[0,0,1]],float)
        self.quat_gy = quaternion.from_rotation_matrix(self.ini_ori)
        #self.quat_gy = quaternion.from_rotation_matrix(np.identity(3,float))

    def time_update(self, T):
        time = T - self.time_t
        self.DT = time/1000
        self.time_t = T

    def filesave(self, data_save):
        inp1 = open("Data1.txt", "a+")
        inp1.write(data_save)
        inp1.write("\n")
        inp1.close()

    def motion_model(self, U):
        T = self.DT
        self.x_states[3:6] = self.x_states[3:6] + (T * (self.orientation.dot(U[3:6]) - np.array([0.0, 0.0, 9.8])))
        self.x_states[0:3] = self.x_states[0:3] + (T * (self.x_states[3:6]))

    def attitude_update(self,U):
        T = self.DT
        """dOmega = np.array([
            [0, -U[2], U[1]],
            [U[2], 0, -U[0]],
            [-U[1], U[0], 0]
        ])"""
        Sw = quaternion.quaternion(0, U[0],U[1],U[2])
        qdot = np.multiply((0.5 * self.quat_gy), Sw)
        quat = np.add(self.quat_gy, (qdot * T))
        quat_arr = quaternion.as_float_array(quat)
        quat_arr = np.divide(quat_arr,math.sqrt((quat_arr[0] ** 2 + quat_arr[1] ** 2 + quat_arr[2] ** 2 + quat_arr[3] ** 2)))
        self.quat_gy = quaternion.quaternion(quat_arr[0], quat_arr[1], quat_arr[2], quat_arr[3])
        self.orientation = quaternion.as_rotation_matrix(self.quat_gy)
        #self.orientation = np.matmul(self.orientation, np.matmul((2 * np.identity(3) + dOmega * T),(np.linalg.inv((2 * np.identity(3) - dOmega * T)))))
        q = quaternion.as_float_array(self.quat_gy)
        yaw = (math.atan2(2.0 * (q[1] * q[2] - q[0] * q[3]), -1 + 2 * (q[0] * q[0] + q[1] * q[1])))
        return yaw

    def error_motion_model(self, U):
        accn = (self.orientation.dot(U[3:6]))
        S = np.array([[0, -accn[2], accn[1]],
                      [accn[2], 0, -accn[0]],
                      [-accn[1], accn[0], 0]])  # The acceleration is bias corrected and transformed to the navigation frame accn = Rot_b_to_n * accn_body

        phi1_1 = np.concatenate((np.identity(3), -self.DT * self.orientation), axis=1)
        phi1 = np.concatenate((phi1_1, np.zeros([3, 9], float)), axis=1)
        phi2_1 = np.concatenate((np.zeros([3, 3], float), np.identity(3)), axis=1)
        phi2 = np.concatenate((phi2_1, np.zeros([3, 9], float)), axis=1)
        phi3_1 = np.concatenate((np.zeros([3, 3], float), np.zeros([3, 3], float)), axis=1)
        phi3_2 = np.concatenate((phi3_1, np.identity(3)), axis=1)
        phi3_3 = np.concatenate((self.DT * np.identity(3), np.zeros([3, 3], float)), axis=1)
        phi3 = np.concatenate((phi3_2, phi3_3), axis=1)
        phi4_1 = np.concatenate((self.DT * S, np.zeros([3, 6], float)), axis=1)
        phi4_2 = np.concatenate((phi4_1, np.identity(3)), axis=1)
        phi4 = np.concatenate((phi4_2, self.DT * self.orientation), axis=1)
        phi5 = np.concatenate((np.zeros([3, 12], float), np.identity(3)), axis=1)
        self.Phi = np.concatenate((phi1, phi2, phi3, phi4, phi5), axis=0)
        self.error_states = self.Phi.dot(self.error_states)
        return self.error_states

    def error_observation_model_acc(self):
        h1 = np.concatenate((np.array([[0, 0, 1]], float), np.zeros([1, 12], float)), axis=1)
        h2 = np.concatenate((np.zeros((3, 9), float), np.identity(3, float), np.zeros((3, 3), float)), axis=1)
        self.H = np.concatenate((h1, h2), axis=0)
        Z = -self.H.dot(self.error_states)
        return Z

    def error_observation_model_gyro(self):
        h1 = np.concatenate((np.array([[0, 0, 1]], float), np.zeros([1, 12], float)), axis=1)
        h2 = np.concatenate((np.zeros((3, 3), float), np.identity(3, float), np.zeros((3, 9), float)), axis=1)
        self.H = np.concatenate((h1, h2), axis=0)
        Z = -self.H.dot(self.error_states)
        return Z

    def error_observation_model_gyro_acc(self):
        h1 = np.concatenate((np.array([[0, 0, 1]], float), np.zeros([1, 12], float)), axis=1)
        h2 = np.concatenate((np.zeros((3, 3), float), np.identity(3, float), np.zeros((3, 9), float)), axis=1)
        h3 = np.concatenate((np.zeros((3, 9), float), np.identity(3, float), np.zeros((3, 3), float)), axis=1)
        self.H = np.concatenate((h1, h2,h3), axis=0)
        Z = -self.H.dot(self.error_states)
        return Z


    def error_observation_model_loc(self):
        h1 = np.concatenate((np.array([[0, 0, 1]], float), np.zeros([1, 12], float)), axis=1)
        h2 = np.concatenate((np.zeros((3, 6), float), np.identity(3, float), np.zeros((3, 6), float)), axis=1)
        self.H = np.concatenate((h1, h2), axis=0)
        Z = -self.H.dot(self.error_states)
        return Z



    def kalman_filter(self, U_vec, acc=False,gyro=False, zk=None, location=None, flag=False):
        try:
            if (not flag):  # Not stationary
                # Prediction
                self.Pk = np.matmul(self.Phi, np.matmul(self.Pk, self.Phi.T)) + self.Qk
            else:  # Stationary
                # Prediction
                if(location):
                    self.error_states = self.error_motion_model(U_vec)
                    Rk = np.diag([0.001, 0.0006, 0.0004, 0.000001])  # --
                    prediction = self.error_observation_model_loc()
                    self.Pk = np.matmul(self.Phi, np.matmul(self.Pk, self.Phi.T)) + self.Qk
                    #Rk = (np.identity(4, float) * np.trace(np.matmul(self.H ,np.matmul(self.Pk , self.H.transpose()))))
                    # Correction
                    vk =  prediction - zk
                    S = (np.matmul(self.H, np.matmul(self.Pk, self.H.T))) + Rk
                    Kk = np.matmul(np.matmul(self.Pk, self.H.T), np.linalg.inv(S))
                    self.error_states = self.error_states + np.matmul(Kk, vk)  # Update estimate with gain * residual
                    tmp = (np.identity(15) - np.matmul(Kk, self.H))
                    self.Pk = np.matmul(np.matmul(tmp, self.Pk),tmp.transpose())+np.matmul(np.matmul(Kk,Rk),Kk.transpose())  # Update error covariance
                    output = self.error_states
                    self.x_states[0:3] = self.x_states[0:3] - output[6:9].T.flatten()
                    self.x_states[3:6] = self.x_states[3:6] - output[9:12].T.flatten()
                    dPhi = self.error_states[0:3]
                    Sw = quaternion.quaternion(0, -dPhi[0], -dPhi[1], -dPhi[2])
                    qdot = np.multiply((0.5 * self.quat_gy), Sw)
                    quat = np.add(self.quat_gy, (qdot ))
                    quat_arr = quaternion.as_float_array(quat)
                    quat_arr = np.divide(quat_arr, math.sqrt((quat_arr[0] ** 2 + quat_arr[1] ** 2 + quat_arr[2] ** 2 + quat_arr[3] ** 2)))
                    self.quat_gy = quaternion.quaternion(quat_arr[0], quat_arr[1], quat_arr[2], quat_arr[3])
                    self.orientation = quaternion.as_rotation_matrix(self.quat_gy)
                    # self.orientation = np.matmul(self.orientation, np.matmul((2 * np.identity(3) + dOmega * T),(np.linalg.inv((2 * np.identity(3) - dOmega * T)))))
                    self.error_states = np.matmul(self.error_states.T,np.diag([0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0])).reshape(-1,1)
                elif (gyro and acc):
                    self.error_states = self.error_motion_model(U_vec)
                    Rk = np.diag([0.001, 0.0001, 0.0001, 0.0001, 0.005,0.005,0.005])  # --
                    self.Pk = np.matmul(self.Phi, np.matmul(self.Pk, self.Phi.T)) + self.Qk
                    prediction = self.error_observation_model_gyro_acc()
                    vk = prediction - zk
                    S = (np.matmul(self.H, np.matmul(self.Pk, self.H.T))) + Rk
                    Kk = np.matmul(np.matmul(self.Pk, self.H.T), np.linalg.inv(S))
                    self.error_states = self.error_states + np.matmul(Kk, vk)  # Update estimate with gain * residual
                    tmp = (np.identity(15) - np.matmul(Kk, self.H))
                    self.Pk = np.matmul(np.matmul(tmp, self.Pk), tmp.transpose()) + np.matmul(np.matmul(Kk, Rk),Kk.transpose())  # Update error covariance
                    output = self.error_states
                    self.x_states[0:3] = self.x_states[0:3] - output[6:9].T.flatten()
                    self.x_states[3:6] = self.x_states[3:6] - output[9:12].T.flatten()
                    dPhi = self.error_states[0:3]
                    Sw = quaternion.quaternion(0, -dPhi[0], -dPhi[1], -dPhi[2])
                    qdot = np.multiply((0.5 * self.quat_gy), Sw)
                    quat = np.add(self.quat_gy, (qdot))
                    quat_arr = quaternion.as_float_array(quat)
                    quat_arr = np.divide(quat_arr, math.sqrt(
                        (quat_arr[0] ** 2 + quat_arr[1] ** 2 + quat_arr[2] ** 2 + quat_arr[3] ** 2)))
                    self.quat_gy = quaternion.quaternion(quat_arr[0], quat_arr[1], quat_arr[2], quat_arr[3])
                    self.orientation = quaternion.as_rotation_matrix(self.quat_gy)
                    self.error_states = np.matmul(self.error_states.T, np.diag([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).reshape(-1, 1)
                elif(gyro and not acc):
                    self.error_states = self.error_motion_model(U_vec)
                    Rk = np.diag([0.001, 0.0001, 0.0001, 0.0001])  # --
                    prediction = self.error_observation_model_gyro()
                    self.Pk = np.matmul(self.Phi, np.matmul(self.Pk, self.Phi.T)) + self.Qk
                    #Rk = (np.identity(4, float) * np.trace(np.matmul(self.H, np.matmul(self.Pk, self.H.transpose()))))
                    # Correction
                    vk = prediction - zk
                    S = (np.matmul(self.H, np.matmul(self.Pk, self.H.T))) + Rk
                    Kk = np.matmul(np.matmul(self.Pk, self.H.T), np.linalg.inv(S))
                    self.error_states = self.error_states + np.matmul(Kk, vk)  # Update estimate with gain * residual
                    tmp = (np.identity(15) - np.matmul(Kk, self.H))
                    self.Pk = np.matmul(np.matmul(tmp, self.Pk),tmp.transpose())+np.matmul(np.matmul(Kk,Rk),Kk.transpose())  # Update error covariance
                    output = self.error_states
                    self.x_states[0:3] = self.x_states[0:3] - output[6:9].T.flatten()
                    self.x_states[3:6] = self.x_states[3:6] - output[9:12].T.flatten()
                    dPhi = self.error_states[0:3]
                    Sw = quaternion.quaternion(0, -dPhi[0], -dPhi[1], -dPhi[2])
                    qdot = np.multiply((0.5 * self.quat_gy), Sw)
                    quat = np.add(self.quat_gy, (qdot))
                    quat_arr = quaternion.as_float_array(quat)
                    quat_arr = np.divide(quat_arr, math.sqrt(
                        (quat_arr[0] ** 2 + quat_arr[1] ** 2 + quat_arr[2] ** 2 + quat_arr[3] ** 2)))
                    self.quat_gy = quaternion.quaternion(quat_arr[0], quat_arr[1], quat_arr[2], quat_arr[3])
                    self.orientation = quaternion.as_rotation_matrix(self.quat_gy)
                    self.error_states = np.matmul(self.error_states.T,np.diag([0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0])).reshape(-1,1)
                elif(acc and not gyro):
                    self.error_states = self.error_motion_model(U_vec)
                    Rk = np.diag([0.001, 0.005, 0.005, 0.005])  # --
                    prediction = self.error_observation_model_acc()
                    self.Pk = np.matmul(self.Phi, np.matmul(self.Pk, self.Phi.T)) + self.Qk
                    #Rk = (np.identity(4, float) * np.trace(np.matmul(self.H, np.matmul(self.Pk, self.H.transpose()))))
                    # Correction
                    vk = prediction - zk
                    S = (np.matmul(self.H, np.matmul(self.Pk, self.H.T))) + Rk
                    Kk = np.matmul(np.matmul(self.Pk, self.H.T), np.linalg.inv(S))
                    self.error_states = self.error_states + np.matmul(Kk, vk)  # Update estimate with gain * residual
                    tmp = (np.identity(15) - np.matmul(Kk, self.H))
                    self.Pk = np.matmul(np.matmul(tmp, self.Pk), tmp.transpose()) + np.matmul(np.matmul(Kk, Rk),
                                                                                              Kk.transpose())  # Update error covariance
                    output = self.error_states
                    self.x_states[0:3] = self.x_states[0:3] - output[6:9].T.flatten()
                    self.x_states[3:6] = self.x_states[3:6] - output[9:12].T.flatten()
                    dPhi = self.error_states[0:3]
                    self.error_states = np.matmul(self.error_states.T, np.diag([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).reshape(-1, 1)
                """else:
                    self.error_states = self.error_motion_model(U_vec)
                    Rk = np.diag([0.01, 0.01, 0.01, 0.01])  # --
                    prediction = self.error_observation_model_stationary_acc()
                    self.Pk = np.matmul(self.Phi, np.matmul(self.Pk, self.Phi.T)) + self.Qk
                    # Correction
                    vk = prediction - zk
                    S = (np.matmul(self.H, np.matmul(self.Pk, self.H.T))) + Rk
                    Kk = np.matmul(np.matmul(self.Pk, self.H.T), np.linalg.inv(S))
                    self.error_states = self.error_states + np.matmul(Kk, vk)  # Update estimate with gain * residual
                    tmp = (np.identity(15) - np.matmul(Kk, self.H))
                    self.Pk = np.matmul(np.matmul(tmp, self.Pk), tmp.transpose()) + np.matmul(np.matmul(Kk, Rk),
                                                                                              Kk.transpose())  # Update error covariance
                    output = self.error_states
                    self.x_states[0:3] = self.x_states[0:3] - output[6:9].T.flatten() * self.DT
                    self.x_states[3:6] = self.x_states[3:6] - output[9:12].T.flatten() * self.DT
                    dPhi = self.error_states[0:3]
                    dTheta = np.array([
                        [0, dPhi[2], -dPhi[1]],
                        [-dPhi[2], 0, dPhi[0]],
                        [dPhi[1], -dPhi[0], 0]
                    ], float)
                    self.orientation = np.matmul(
                        np.matmul((2 * np.identity(3) + dTheta), np.linalg.inv(2 * np.identity(3) - dTheta)),
                        self.orientation)
                    self.error_states = np.matmul(self.error_states.T, np.diag(
                        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).reshape(-1, 1)"""

            #print( self.x_states[0:3])
        except np.linalg.linalg.LinAlgError:
            pass

    def main(self):
        f = open('C:\\Users\\smchakra\\Desktop\\Experiments\\Experiments_Python\\Parking Lot\\03_17\\Walking_Right\\Tri_XYZ\\Nav_Walking_Right.txt', 'r')
        line = f.readline()
        self.time_t = float(line.split(",")[0])
        magr = np.array([float(line.split(",")[7]), float(line.split(",")[8]), float(line.split(",")[9])])
        accn = np.array([float(line.split(",")[1]), float(line.split(",")[2]), float(line.split(",")[3])])
        data = np.concatenate((accn,magr),axis=0)
        self.first_init(data,line.split(",")[0])
        sensr = sensor_fusion(self.ini_ori, self.time_t)
        hxEst = np.array([0.0,0.0,0.0]).reshape(-1,1)
        still_time = 0.0

        while line:
                line = f.readline()
                line = line.strip()
                splitted = line.split(",")
                self.time_update(float(line.split(",")[0]))
                magr = (np.array([float(line.split(",")[7]), float(line.split(",")[8]), float(line.split(",")[9])]).reshape(-1, 1))
                accn = 0.6*accn.reshape(-1,1)+0.4*((np.array([float(line.split(",")[1]), float(line.split(",")[2]), float(line.split(",")[3])]).reshape(-1,1) - np.array([self.error_states[12][0], self.error_states[13][0], self.error_states[14][0]]).reshape(-1, 1)))
                gyro = (np.array([math.radians(float(line.split(",")[4])), math.radians(float(line.split(",")[5])),math.radians(float(line.split(",")[6]))]).reshape(-1, 1) - np.array([self.error_states[3][0], self.error_states[4][0], self.error_states[5][0]]).reshape(-1, 1))
                U_vec = np.concatenate((gyro.T, accn.T), axis=1).flatten()
                yaw = self.attitude_update(U_vec)
                self.motion_model(U_vec)
                sensr.set_angles(np.array([float(line.split(",")[1]), float(line.split(",")[2]), float(line.split(",")[3])]).reshape(-1,1), magr, self.DT)
                print(yaw)
                if (len(splitted)==22):
                    z = np.zeros((4,1),float)
                    style = "x"
                    z[0][0] = sensr.yaw_a - yaw- math.radians(2)
                    loc = np.array([np.array([float(splitted[10]), float(splitted[11]),0.0])- self.x_states[0:3]])
                    z[1][0] = loc[0,0]
                    z[2][0] = loc[0,1]
                    z[3][0] = loc[0,2]
                    self.kalman_filter(U_vec=U_vec, zk=z, location=True, flag=True)
                elif(8.8<(abs(np.linalg.norm(np.array([float(line.split(",")[1]), float(line.split(",")[2]), float(line.split(",")[3])]))))<11.2and abs(np.linalg.norm(np.array([float(line.split(",")[4]), float(line.split(",")[5]), float(line.split(",")[6])])))>45 ):
                    z = np.zeros((4,1), float)
                    style = "o"
                    z[0][0] =yaw-sensr.yaw_a- math.radians(2)
                    z[1][0] = -self.x_states[3]
                    z[2][0] = -self.x_states[4]
                    z[3][0] = -self.x_states[5]
                    self.kalman_filter(U_vec=U_vec,acc=True,gyro=False, zk=z, location=False, flag=True)
                elif (abs(np.linalg.norm(np.array([float(line.split(",")[4]), float(line.split(",")[5]), float(line.split(",")[6])]))) < 45 and (8.8>(abs(np.linalg.norm(np.array([float(line.split(",")[1]), float(line.split(",")[2]), float(line.split(",")[3])])))) or (abs(np.linalg.norm(np.array([float(line.split(",")[1]), float(line.split(",")[2]), float(line.split(",")[3])]))))>11.2)):
                    z = np.zeros((4, 1), float)
                    style = "o"
                    z[0][0] =yaw-sensr.yaw_a - math.radians(2)
                    z[1][0] =  -math.radians(float(line.split(",")[4]))
                    z[2][0] =  -math.radians(float(line.split(",")[5]))
                    z[3][0] = -math.radians(float(line.split(",")[6]))
                    self.kalman_filter(U_vec=U_vec, acc=False,gyro=True,zk=z, location=False, flag=True)
                elif (abs(np.linalg.norm(np.array([float(line.split(",")[4]), float(line.split(",")[5]),float(line.split(",")[6])]))) <45 and 8.8<abs(abs(np.linalg.norm(np.array([float(line.split(",")[1]), float(line.split(",")[2]), float(line.split(",")[3])])))) <11.2):
                    z = np.zeros((7, 1), float)
                    style = "o"
                    z[0][0] =yaw-sensr.yaw_a- math.radians(2)
                    z[1][0] = -math.radians(float(line.split(",")[4]))
                    z[2][0] =- math.radians(float(line.split(",")[5]))
                    z[3][0] = -math.radians(float(line.split(",")[6]))
                    z[4][0] = -self.x_states[3]
                    z[5][0] = -self.x_states[4]
                    z[6][0] = -self.x_states[5]
                    self.kalman_filter(U_vec=U_vec, acc=True,gyro=True, zk=z, location=False, flag=True)
                else:
                    self.kalman_filter(U_vec=U_vec,gyro=False, zk=None, location=None, flag=False)
                    style = "o"
                hxEst = np.hstack((hxEst, self.x_states[0:3].reshape(-1,1)))
                plt.cla()
                plt.scatter(hxEst[0, :].flatten(), hxEst[1, :].flatten(),marker=style)
                plt.plot(hxEst[0, :].flatten(), hxEst[1, :].flatten(), "-k")
                plt.axis("equal")
                plt.grid(True)
                plt.pause(0.001)


obj = main_Class()
ini_time = time.time()
obj.main()

import socket
import math
import random
import json
import numpy as np
import time
from scipy import signal
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from new_own_trial import sensor_fusion
from sklearn.preprocessing import normalize

class main_Class(object):
    def __init__(self):
        self.Phi = np.identity(15, float)
        self.B = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], float)
        self.H = np.zeros((10, 9), float)   #--
        self.Pk = np.diag([0, 0, 0, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0, 0, 0, 0, 0.01, 0.01, 0.01])
        arr = np.concatenate((np.array([[0.0001, 0.0001, 0.0001]]), np.zeros((1, 6), float),np.array([[0.0001, 0.0001, 0.0001]]), np.zeros((1, 3), float)), axis=1)
        self.Qk = np.diag(arr)
        self.calib_result = np.array([0, 0, 0], float)
        self.ini_ori = np.identity(3)
        self.ini_ori2 = np.identity(3)
        self.orientation = self.ini_ori
        self.x_states = np.array([0, 0, 0, 0, 0, 0], float)
        self.error_states = np.zeros((15, 1), float)
        self.U = np.array([[0, 0, 0, 0, 0, 0]], float).T
        self.time_t = 0.0
        self.DT = 0.0
        self.quat_gy = Quaternion(array=[0, 0, 0, 0])
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
        self.quat_gy = Quaternion(matrix=np.matmul(tmp,self.ini_ori))

    def time_update(self, T):
        time = T - self.time_t
        self.DT = time/1000
        self.time_t = T

    def filesave(self, data_save):
        # data_save = data_save.strip()
        # data_save = str(data_save).strip('[ ]')
        # data_save = data_save.strip("' '")
        inp1 = open("Data1.txt", "a+")
        inp1.write(data_save)
        inp1.write("\n")
        inp1.close()

    def motion_model(self, U):
        T = self.DT
        dOmega = np.array([
            [0, -U[2], U[1]],
           [U[2], 0, -U[0]],
          [-U[1], U[0], 0]
        ])
        S = Quaternion(scalar=0.0, vector=[U[0], U[1], U[2]])
        qdot = (0.5 * (self.quat_gy * S))
        quat = self.quat_gy + (qdot * T)
        self.quat_gy = quat.normalised
        self.orientation = self.quat_gy.rotation_matrix
        #self.orientation = np.matmul(self.orientation,np.matmul((2 * np.identity(3) + dOmega * T),(np.linalg.inv( (2 * np.identity(3) - dOmega * T)))))
        self.x_states[3:6] = self.x_states[3:6] + (T * (self.orientation.dot(U[3:6]) - np.array([0.0, 0.0, 9.8])))
        self.x_states[0:3] = self.x_states[0:3] + (T * (self.x_states[3:6]))

    def error_motion_model(self, U):
        accn = (self.orientation.dot(U[3:6]))
        S = -np.array([[0, -accn[2], accn[1]],
                      [accn[2], 0, -accn[0]],
                      [-accn[1], accn[0], 0]])  # The acceleration is bias corrected and transformed to the navigation frame accn = Rot_b_to_n * accn_body

        phi1_1 = np.concatenate((np.identity(3), self.DT * self.orientation), axis=1)
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
        Z = self.H.dot(self.error_states)
        return Z

    def error_observation_model_gyro(self):
        h1 = np.concatenate((np.array([[0, 0, 1]], float), np.zeros([1, 12], float)), axis=1)
        h2 = np.concatenate((np.zeros((3, 3), float), np.identity(3, float), np.zeros((3, 9), float)), axis=1)
        self.H = np.concatenate((h1, h2), axis=0)
        Z = self.H.dot(self.error_states)
        return Z

    def error_observation_model_gyro_acc(self):
        h1 = np.concatenate((np.array([[0, 0, 1]], float), np.zeros([1, 12], float)), axis=1)
        h2 = np.concatenate((np.zeros((3, 3), float), np.identity(3, float), np.zeros((3, 9), float)), axis=1)
        h3 = np.concatenate((np.zeros((3, 9), float), np.identity(3, float), np.zeros((3, 3), float)), axis=1)
        self.H = np.concatenate((h1, h2,h3), axis=0)
        Z = self.H.dot(self.error_states)
        return Z


    def error_observation_model_loc(self):
        h1 = np.concatenate((np.array([[0, 0, 1]], float), np.zeros([1, 12], float)), axis=1)
        h2 = np.concatenate((np.zeros((3, 6), float), np.identity(3, float), np.zeros((3, 6), float)), axis=1)
        self.H = np.concatenate((h1, h2), axis=0)
        Z = self.H.dot(self.error_states)
        return Z

    def get_heading(self):
        quat = Quaternion(matrix=self.orientation)
        q = quat.normalised
        q = q.elements
        yaw = (math.atan2(2.0 * (q[1] * q[2] - q[0] * q[3]),
                          -1 + 2 * (q[0] * q[0] + q[1] * q[1])))
        return yaw

    def heading_from_magnetometer(self,magn):
        phi_k = math.atan2(self.orientation[2][1], self.orientation[2][2])
        theta_k = math.asin(self.orientation[2][0])
        mat1 = np.array([[math.cos(theta_k),0,-math.sin(theta_k)],[0,1,0],[-math.sin(theta_k), 0, math.cos(theta_k)]])
        mat2 = np.array([[1,0,0],[0,-math.cos(phi_k),-math.sin(phi_k)],[0,math.sin(phi_k),math.cos(phi_k)]])
        mat = np.matmul(np.matmul(mat1,mat2),magn)
        yaw = -math.atan2(mat[1],mat[0]) - math.radians(2)
        return yaw
    def kalman_filter(self, U_vec, acc=False,gyro=False, zk=None, location=None, flag=False):
        try:
            if (not flag):  # Not stationary
                # Prediction
                self.Pk = np.matmul(self.Phi, np.matmul(self.Pk, self.Phi.T)) + self.Qk
            else:  # Stationary
                # Prediction
                if(location):
                    self.error_states = self.error_motion_model(U_vec)
                    Rk = np.diag([0.01, 0.01, 0.01, 0.01])  # --
                    prediction = self.error_observation_model_loc()
                    self.Pk = np.matmul(self.Phi, np.matmul(self.Pk, self.Phi.T)) + self.Qk
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
                    dTheta = np.array([
                        [0, dPhi[2], -dPhi[1]],
                        [-dPhi[2], 0, dPhi[0]],
                        [dPhi[1], -dPhi[0], 0]
                    ], float)
                    self.orientation = np.matmul(np.matmul((2 * np.identity(3) + dTheta), np.linalg.inv(2 * np.identity(3) - dTheta)),self.orientation)
                    self.error_states = np.matmul(self.error_states.T,np.diag([0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0])).reshape(-1,1)
                elif (gyro and acc):
                    self.error_states = self.error_motion_model(U_vec)
                    Rk = np.diag([0.01, 0.01, 0.01, 0.01, 0.1,0.1,0.1])  # --
                    prediction = self.error_observation_model_gyro_acc()
                    self.Pk = np.matmul(self.Phi, np.matmul(self.Pk, self.Phi.T)) + self.Qk
                    # Correction
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
                    dTheta = np.array([
                        [0, dPhi[2], -dPhi[1]],
                        [-dPhi[2], 0, dPhi[0]],
                        [dPhi[1], -dPhi[0], 0]
                    ], float)
                    self.orientation = np.matmul(
                        np.matmul((2 * np.identity(3) + dTheta), np.linalg.inv(2 * np.identity(3) - dTheta)),
                        self.orientation)
                    self.error_states = np.matmul(self.error_states.T, np.diag(
                        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).reshape(-1, 1)
                elif(gyro and not acc):
                    self.error_states = self.error_motion_model(U_vec)
                    Rk = np.diag([0.01, 0.01, 0.01, 0.01])  # --
                    prediction = self.error_observation_model_gyro()
                    self.Pk = np.matmul(self.Phi, np.matmul(self.Pk, self.Phi.T)) + self.Qk
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
                    dTheta = np.array([
                        [0, dPhi[2], -dPhi[1]],
                        [-dPhi[2], 0, dPhi[0]],
                        [dPhi[1], -dPhi[0], 0]
                    ], float)
                    self.orientation = np.matmul(
                        np.matmul((2 * np.identity(3) + dTheta), np.linalg.inv(2 * np.identity(3) - dTheta)),
                        self.orientation)
                    self.error_states = np.matmul(self.error_states.T,np.diag([0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0])).reshape(-1,1)
                elif(acc and not gyro):
                    self.error_states = self.error_motion_model(U_vec)
                    Rk = np.diag([0.01, 0.1, 0.1, 0.1])  # --
                    prediction = self.error_observation_model_acc()
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
                    self.x_states[0:3] = self.x_states[0:3] - output[6:9].T.flatten()
                    self.x_states[3:6] = self.x_states[3:6] - output[9:12].T.flatten()
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
                        [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])).reshape(-1, 1)
                tmp_quat = Quaternion(matrix=self.orientation)
                #self.orientation = tmp_quat.normalised.rotation_matrix
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
            print( self.x_states[0:3])
        except np.linalg.linalg.LinAlgError:
            pass

    def main(self):
        sensr = sensor_fusion(self.ini_ori, self.time_t)
        f = open('C:\\Users\\smchakra\\Desktop\\Experiments\\Experiments_Python\\Parking Lot\\03_17\\walking_right_copy_xyz.txt', 'r')
        line = f.readline()
        self.time_t = float(line.split(",")[0])
        magr = np.array([float(line.split(",")[7]), float(line.split(",")[8]), float(line.split(",")[9])])
        accn = np.array([float(line.split(",")[1]), float(line.split(",")[2]), float(line.split(",")[3])])
        data = np.concatenate((accn,magr),axis=0)
        self.first_init(data,line.split(",")[0])
        #self.ini_ori2 = self.ini_ori
        self.ini_ori2= np.array([[-1,0,0],[0,1,0],[0,0,-1]],float)
        hxEst = np.array([0.0,0.0,0.0])
        while line:
                line = f.readline()
                line = line.strip()
                splitted = line.split(",")
                self.time_update(float(line.split(",")[0]))
                magr = (np.array([float(line.split(",")[7]), float(line.split(",")[8]), float(line.split(",")[9])]).reshape(-1, 1))
                accn = (np.array([float(line.split(",")[1]), float(line.split(",")[2]), float(line.split(",")[3])]).reshape(-1,1) - np.array([self.error_states[12][0], self.error_states[13][0], self.error_states[14][0]]).reshape(-1, 1))
                gyro = (np.array([math.radians(float(line.split(",")[4])), math.radians(float(line.split(",")[5])),math.radians(float(line.split(",")[6]))]).reshape(-1, 1) - np.array([self.error_states[3][0], self.error_states[4][0], self.error_states[5][0]]).reshape(-1, 1))
                U_vec = np.concatenate((gyro.T, accn.T), axis=1).flatten()
                self.motion_model(U_vec)
                sensr.set_angles(np.array([float(line.split(",")[1]), float(line.split(",")[2]), float(line.split(",")[3])]).reshape(-1,1), magr, self.DT)
                yaw = self.get_heading()
                yaw_new = self.heading_from_magnetometer(magr)
                if (len(splitted)==22):
                    print("Location")
                    z = np.zeros((4,1),float)
                    z[0][0] = yaw - sensr.yaw_a
                    loc = np.array([self.x_states[0:3] - np.array([float(splitted[10]), float(splitted[11]),0.0])])
                    z[1][0] = loc[0,0]
                    z[2][0] = loc[0,1]
                    z[3][0] = loc[0,2]
                    self.kalman_filter(U_vec=U_vec, zk=z, location=True, flag=True)
                elif(abs(11-abs(np.linalg.norm(np.array([float(line.split(",")[1]), float(line.split(",")[2]), float(line.split(",")[3])]))))<2and abs(np.linalg.norm(np.array([float(line.split(",")[4]), float(line.split(",")[5]), float(line.split(",")[6])])))>25 ):
                    print("Acc ")
                    z = np.zeros((4,1), float)
                    z[0][0] = yaw - sensr.yaw_a
                    z[1][0] = self.x_states[3]
                    z[2][0] = self.x_states[4]
                    z[3][0] = self.x_states[5]
                    self.kalman_filter(U_vec=U_vec,acc=True,gyro=False, zk=z, location=False, flag=True)
                elif (abs(np.linalg.norm(np.array([float(line.split(",")[4]), float(line.split(",")[5]), float(line.split(",")[6])]))) < 25 and abs(11-abs(np.linalg.norm(np.array([float(line.split(",")[1]), float(line.split(",")[2]), float(line.split(",")[3])]))))>2 ):
                    print("gyro")
                    z = np.zeros((4, 1), float)
                    z[0][0] = yaw - sensr.yaw_a
                    z[1][0] = math.radians(float(line.split(",")[4]))
                    z[2][0] = math.radians(float(line.split(",")[5]))
                    z[3][0] = math.radians(float(line.split(",")[6]))
                    self.kalman_filter(U_vec=U_vec, acc=False,gyro=True,zk=z, location=False, flag=True)
                elif (abs(np.linalg.norm(np.array([float(line.split(",")[4]), float(line.split(",")[5]),float(line.split(",")[6])]))) < 25 and abs(11 - abs(np.linalg.norm(np.array([float(line.split(",")[1]), float(line.split(",")[2]), float(line.split(",")[3])])))) < 2):
                    print("acc and gyro")
                    z = np.zeros((7, 1), float)
                    z[0][0] = yaw - sensr.yaw_a
                    z[1][0] = math.radians(float(line.split(",")[4]))
                    z[2][0] = math.radians(float(line.split(",")[5]))
                    z[3][0] = math.radians(float(line.split(",")[6]))
                    z[4][0] = self.x_states[3]
                    z[5][0] = self.x_states[4]
                    z[6][0] = self.x_states[5]
                    self.kalman_filter(U_vec=U_vec, acc=True,gyro=True, zk=z, location=False, flag=True)
                else:
                    print("None")
                    self.kalman_filter(U_vec=U_vec,gyro=False, zk=None, location=None, flag=False)


obj = main_Class()
ini_time = time.time()
obj.main()

#!/usr/bin/env python3

import rospy
import socket
import numpy as np
from latis_msgs.msg import raw_radar
import time

class udp_receiver():

    def __init__(self):
        # Set our class attributes (persistant variables)
        self.udp_port = 12345
        self.buffer_size = 2048

        # Define a socket listener to receive the radar packets
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def main_loop(self):
        #  Open UDP Socket
        self.sock.bind(('', self.udp_port))
        self.sock.setblocking(True)
        print('UDP socket 12345 opened.')

        #  Read in Data
        while True:

            byteArray, addr = self.sock.recvfrom(self.buffer_size)
            # print(byteArray)
            info = str(byteArray)

            one = info.find(" 1, ")
            three = info.find(" 3, ")
            four = info.find(" 4, ")
            five = info.find(" 5, ")
            #
            # print("Found ', 1,' at:", one)
            # print("Found ', 3,' at:", three)
            # print("Found ', 4,' at:", four)
            # print("Found ', 5,' at:", five)
            if one != -1:
                gps = info[one:three-2].split(',')
                print("data:", gps)
                lat = float(gps[1])
                lon = float(gps[2])
                alt = float(gps[3])
                print("lat:", lat)
                print("lon:", lon)
                print("alt:", alt)
                # accel = info[three:four-2].split(',')
                # print("Accel:", accel)
                # x_accel = float(accel[1])
                # y_accel = float(accel[2])
                # z_accel = float(accel[3])
                # print("x accel:", x_accel)
                # print("y accel:", y_accel)
                # print("z accel:", z_accel)
                # gyro = info[four:five-2].split(',')
                # print("Gyro:", gyro)
                # x_gyro = float(gyro[1])
                # y_gyro = float(gyro[2])
                # z_gyro = float(gyro[3])
                # print("x gyro:", x_gyro)
                # print("y gyro:", y_gyro)
                # print("z gyro:", z_gyro)
            if three != -1:
                accel = info[three:four-2].split(',')
                print("Accel:", accel)
                x_accel = float(accel[1])
                y_accel = float(accel[2])
                z_accel = float(accel[3])
                print("x accel:", x_accel)
                print("y accel:", y_accel)
                print("z accel:", z_accel)
            if four != -1:
                gyro = info[four:five-2].split(',')
                print("Gyro:", gyro)
                x_gyro = float(gyro[1])
                y_gyro = float(gyro[2])
                z_gyro = float(gyro[3])
                print("x gyro:", x_gyro)
                print("y gyro:", y_gyro)
                print("z gyro:", z_gyro)


    def run_receiver(self):
        self.main_loop()

if __name__ == '__main__':
    receiver = udp_receiver()
    receiver.run_receiver()

clear
clc
close all

%Time properties
Ts = 0.5;           %s
T1 = 5;             %s
T2 = 25;            %s
T3 = 30;            %s
Tf = 50;            %s

%System properties
m = 100;            %kg
b = 20;             %N-s/m
sigma_meas = 0.001; %m^2
sigma_vel = 0.01;   %m^2/s^2
sigma_pos = 0.0001; %m^2

%Continous SS
A = [0 1;
     0 -b/m];
 
B = [0;
     1/m];

C = eye(2);

D = zeros(2,1);

sys = ss(A,B,C,D);

sysd = c2d(sys,Ts);



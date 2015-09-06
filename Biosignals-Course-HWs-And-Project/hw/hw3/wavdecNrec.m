clear; clc; close all;
sr=200;
fo=10;
f1=20;
f2=40;
f3=80;
t=0:1/sr:4;
x=cos(2*pi*fo*t)+cos(2*pi*f1*t)+cos(2*pi*f2*t)+cos(2*pi*f3*t);
lx=length(x);
dbname='db6';
N=3;
[C,L]=wavedec(x,N,dbname);

y=waverec(C,L,dbname);
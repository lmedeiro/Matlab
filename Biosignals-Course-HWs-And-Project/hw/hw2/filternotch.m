function [ x ] = filternotch(wnotch,r, x )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
b=[1 -2*cos(wnotch) 1];
a=[1 -2*r*cos(wnotch) r*r];
x=filter(b,a,x);
end


% Assignment 1. Question 2.b. 
% Load 'edbe0103.mat' and plot channel 1 of the signal
clear;close all;clc;
load('edbe0103.mat')

x=sig(:,1); 
x=x-mean(x); %This smoothes the signal. 
figure(1)
subplot(211),plot(x);title('Original EKG signal')
%use sample numbers selected by observation to mark an epoch
n1=222838; n2=223075;
tx=x(n1:n2);
ltx=length(tx);
figure(2)
subplot(311),plot(tx), title('A select epoch to be used as a template') 
% FIR filter with impulse response = tx flipped is 238 samples long
b=flipud(tx); a=1;
figure(2)
subplot(312), plot(b), title('TIme reverse of the template shifter for causality')
rtx=xcorr(tx);      %Autocorrelation of the template
ctx=conv(tx,b);     %Convolution of the template with its time reverse. The result should be the same as rtx.
subplot(313), plot(-ltx+1:ltx-1,rtx), hold,plot(-ltx+1:ltx-1,ctx,'r'), legend('autocorrealtion', 'convolution')
%filter the EKG data with the correlation filter
xx=filter(b,a,x);
figure(1)
subplot(212), plot(xx), title('Correlation filter output') 
 
figure(3) %ZOOM IN around the area where the template came from
subplot(211), plot(x),axis([222838-1500, 223075+1500, -2,4]), title('Original EKG, zoomed in around the template location') 
subplot(212), plot(xx),axis([222838-1500, 223075+1500, -20,30]), hold, pause, plot(n1:n1+length(rtx)-1,rtx,'r'), title('Filtered EKG, zoomed in around the template location, superimposed with rtx in situ') 

figure
subplot(211), plot(x),axis([4.7e5, 4.8e5, -3,3]), title('Original signal zoomed to a region of abnormality')
subplot(212), plot(xx),axis([4.7e5, 4.8e5, -20,30]),title('Filter EKG signal zoomed to a region of abnormality');
%Handpicked three regions on xx, superimposed with rtx.
subplot(212),hold, plot(470753:471227, rtx,'r'),plot(471839:472313,rtx,'r'),plot(472973:473447,rtx,'r')
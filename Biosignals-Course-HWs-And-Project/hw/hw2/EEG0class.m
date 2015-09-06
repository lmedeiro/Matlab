clear; clc; close all;
load val13
%load('edbe0103.mat')
sr=256;
sp=1/sr %sampling period
val=val13';
clear val13
nfft=2^(fix(log2(size(val,1)))+2);
fval=fft(val,nfft);
afval=abs(fval(1:nfft/2+1,:));
t=[0:size(val,1)-1]*sp;
f=[0:nfft/2]/nfft*sr;
figure(1)
for k=1:12
    subplot(12,1,k),plot(t,val(:,k))
    %     axis([250,350,-1000,1000])
end
figure(2)
for k=13:23
    subplot(11,1,k-12),plot(t,val(:,k))
    %     axis([100,400,-1000,1000])
end
x=val(:,4);
clear val;
figure(3)
subplot(211), plot(t,x)
tfr=3; % seconds
[s, hoptime]=frames(x,tfr*1e3,0.8,sr);
nfr=tfr*sr; % samples per frame
nof=size(s,2);
nfft=2^(fix(log2(nfr))+2);
tf=[1:nof]*hoptime;
fs=fft(s,nfft);
afs=abs(fs(1:nfft/2+1,:));
f=[0:nfft/2]*sr/nfft;
figure(4)
imagesc(tf,f,20*log10(afs)), axis xy; colormap(jet);
r=0.98;
for m=1:7
    wnotch=m*pi/8;
    x=filternotch(wnotch,r,x);
    % divide into frames that are
    [s,hoptime]=frames(x,tfr*1e3,0.8,sr);
    
    fs=fft(s,nfft);
    afs=abs(fs(1:nfft/2+1,:));
    f=[0:nfft/2]*sr/nfft;
    figure(4+m)
    imagesc(tf,f,20*log10(afs)), axis xy; colormap(jet);
end
figure(3)
subplot(212), plot(t,x)
figure
for k=1:300 %nof
    plot(f,afs(:,k))
    pause(0.1)
end
figure
for k=1:300 %nof
    plot(f(1:nfft/4),afs(1:nfft/4,k))
    pause(0.1)
end

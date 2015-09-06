clear; clc; close all;
sr=200;
fCounter=1; % counter for figure
fo=10;
f1=20;
f2=40;
f3=80;
t=0:1/sr:4;
x=cos(2*pi*fo*t)+cos(2*pi*f1*t)+cos(2*pi*f2*t)+cos(2*pi*f3*t);
nfft=1024;
fx=fft(x,nfft);
afx=abs(fx(1:nfft/2+1));
f=(0:nfft/2)/nfft*sr;
figure(fCounter);fCounter=fCounter+1;
subplot(211),plot(t,x)
subplot(212),plot(f,afx)
dbname='db6';
N=3;
[C,L]=wavedec(x,N,dbname);
begin=1;
figure(fCounter);fCounter=fCounter+1;
for k=1:N+1
    fin=begin+L(k)-1;
    wc(k)={C(begin:fin)};
    d=cell2mat(wc(k));
    fd=fft(d,nfft);
    afd=abs(fd(1:nfft/2+1));
    % time(k)={[0:
    begin=fin+1;
    if k<2
        p=N
    else
        p=p-1;
    end
    t=(0:L(k)-1)/sr/2^p;
    f=(0:nfft/2)/nfft*sr/2^p;
    subplot(N+1,2,2*k-1),plot(t,d)
    subplot(N+1,2,2*k),plot(f,afd)
end
%start reconstructing
% coef=
% for k=N+1:-1:1
%     d=cell2mat(wc(k));
%     
%     y=upcoeff
%     
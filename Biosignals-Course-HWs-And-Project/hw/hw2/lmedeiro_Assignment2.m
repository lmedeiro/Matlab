%% Assignment 2, Luiz Medeiros;
%  

%%Exercise 1
clc,clear all,close all;
%a. 
Td=[1 2 4 8 16]; % final sampling time in seconds;
sr=[4 8 16];    % sampling rate;
f0=1000;
Fs=[];
Ts=[];
t=[];   % will serve to store our final time frames based on the 
        % above evaluations;
x=[];
strTd='';
strTs='';
strF='';
lt=[];
for k=1:5
    strTd=num2str(Td(k));
    figure(k);
    for n=1:3
        Fs(n)=sr(n)*f0;
        Ts(n)=1/Fs(n);
        t=0:Ts(n):Td(k)-Ts(n);
        x=cos(2*pi*f0*t);
        strTs=num2str(Ts(n));
        strF=strcat('Td= ',strTd,',','Ts= ',strTs);
        lt=size(find(t<0.003)); % This will provide me all the samples before 3 ms. 
        
%         subplot(3,1,n),plot(x(1,1:f0*Td(k)*0.00225)),title(strF);
        subplot(3,1,n),plot(t(1,1:lt(1,2)),x(1,1:lt(1,2))),title(strF); % only outputting the samples up to 3ms.
        
        
    end
        
end

for k=1:5
    strTd=num2str(Td(k));
    figure(5+k);
    for n=1:3
        Fs(n)=sr(n)*f0;
        Ts(n)=1/Fs(n);
        t=0:Ts(n):Td(k)-Ts(n);
        x=cos(2*pi*f0*t);
        strTs=num2str(Ts(n));
        strF=strcat('Td= ',strTd,',','Ts= ',strTs);
        lt=size(find(t<0.003)); % This will provide me all the samples before 3 ms. 
        
%         subplot(3,1,n),plot(x(1,1:f0*Td(k)*0.00225)),title(strF);
        subplot(3,1,n),stem(t(1,1:lt(1,2)),x(1,1:lt(1,2))),title(strF); % only outputting the samples up to 3ms.
        
        
    end
    
    
end

%1.b. 
nfft=512;   % 2^(fix((log2(xSize+2)))
fs=Fs(3);fss=[];
ts=1/fs;
fx=[];
fxx=[];
x=[];t=[];tt=[];
figure;
for k=1:5
    strTd=num2str(Td(k));
    t=0:ts:Td(k)-ts;
    x=cos(2*pi*f0*t);
    fx=fft(x,nfft);
    fxx=abs(fx(1:nfft/2+1));
    fss=[0:nfft/2]*fs/nfft;
    strF=strcat('Td= ',strTd,';');
    subplot(5,1,k),plot(fss,fxx),title(strF); % only outputting the samples up to 3ms. 



end

%% c. 

nfft=[16 26 64 128];
fs=16*f0;
ts=1/fs;
fx=[];
fxx=[];
x=[];t=[];tt=[];om=[];
ix=[];
figure(7),figure(8);
tix=[];
strnfft='';
for k=1:4
    strnfft=num2str(nfft(k));
    t=0:ts:Td(3)-ts;
    x=cos(2*pi*f0*t);
    fx=fft(x,nfft(k));
    ix=ifft(fx,nfft(k));
    tix=[0:length(ix)-1]/ts;
    fxx=abs(fx(1:nfft(k)/2+1));
    fss=[0:nfft(k)/2]*fs/nfft(k);
    om=2*pi*fss/fs;
    strF=strcat('nfft= ',strnfft,';');
    figure(7),subplot(4,1,k),plot(fss,fxx),title(strF),xlabel('x=frequency'); % only outputting the samples up to 3ms. 
    
    figure(8),subplot(4,1,k),plot(tix,ix),title(strF),xlabel('x=time');
    figure(9),subplot(4,1,k),plot(om,fxx),title(strF),xlabel('x=Omega');

end

%% d,e; beginning work on the the window portion. 
% 
clc,clear all,close all;
f0=1000;
fs=16*f0;
ts=1/fs;
Td=1;
nfft=128;
strR='Rectangular';
strH='Hamming';
strB='Barlett';
t=0:ts:Td-ts;
tw=0:1/nfft:Td-1/nfft;
lt=length(t);
x=cos(2*pi*f0*t)+cos(2*pi*3000*t);
w=[window(@rectwin,nfft) window(@hamming,nfft) window(@bartlett,nfft) ];
w=w';
fw=[];
ww=[];
fxw=[];
fAbsXw=[];
a=length(x)-128;
z=zeros(1,a);
xw=[];
for k=1:3
    ww(k,:)=[w(k,:) z];
    xw(k,:)=ww(k,:).*x;
    fxw(k,:)=fft(xw(k,:));
    fAbsXw(k,:)=abs(fxw(k,:));
end

wAbs=[];
for k=1:3
    fw(k,:)=fft(w(k,:),nfft);
    wAbs(k,:)=abs(fw(k,:));  
end
f=(0:nfft/2-1)*2*pi/nfft;
figure(1),plot(tw,w),legend(strR,strH,strB); % plotting against time; 
figure(2),plot(f,20*log10(wAbs(:,1:length(f)))),legend(strR,strH,strB);
figure(3),plot(t(1,1:128),xw(:,1:length(tw))),legend(strR,strH,strB);
figure(4),plot(t,xw),legend(strR,strH,strB);
% fwx=(0:length(fAbsXw(1,:))/2-1)*2*pi*1/length(fAbsXw(1,:));
fwx=(0:length(fAbsXw(1,:))/2-1)*2*pi*ts;
figure(5),plot(fwx,fAbsXw(:,1:length(fwx))),legend(strR,strH,strB),title('mag(XW)  vs OMEGA');
figure(5),plot(fwx,20*log10(fAbsXw(:,1:length(fwx)))),legend(strR,strH,strB),title('mag(XW) dB vs OMEGA');

col='brgk';
zoom xon;

%% Exercise 2;
clear;close all;clc;

load('chb01_02_edfm.mat');
% sr=256;
x=val(7,:);
nfft=2^(fix(log2(length(x)))+2);
figure,plot(x),title('x vs sample');
clear val;
X=fft(x,nfft);

absX=abs(X);
% From the information file, we know that the 
% sampling frequency is 256Hz.
% In addition, we also know that the duration is 10 minutes;(0:10);
fs=256;
ts=1/fs;
td=ts*length(x); % the duration of the siganl equals to the period of each sample times the total number of samples in the signal;
t=0:ts:td-ts;
figure,plot(t,x),title('x vs time t');
f=(0:nfft/2-1)*fs/nfft;
figure, plot(f,absX(1,1:nfft/2)),ylabel('mag(X)'),xlabel('frequency f'),title('mag of X vs f without filtering;');
om=2*pi*f/fs;
figure, plot(om,20*log10(absX(1,1:nfft/2))),ylabel('mag(X) dB'),xlabel('frequency f'),title('mag (x) dB vs Omega Without Filtering');
% After using the spectral tip to determine the frequencies to take away
% from our signal, we will define them here an begin the filtering. 
w1=16;w2=32;w3=44;w4=48;w5=60;w6=64;w7=76;w8=80;w9=96;w10=112;
w=[w1 w2 w3 w4 w5 w6 w7 w8 w9 w10];
w=w/fs*2*pi;
% after we have all of our frequencies decided, we begin by devising our
% filters based on that information; 
r=0.99;
H=[zeros(10,nfft/2)];
w0=[zeros(1,10)];
a=[zeros(10,3)];
b=[zeros(10,3)];
for k=1:10
    w0=w(1,k);
    b(k,:)=[1 -2*cos(w0) 1];
    a(k,:)=[1 -2*r*cos(w0) r^2];
    H(k,:)=freqz(b(k,:),a(k,:),nfft/2);% H represents our filter function; 
    
end
figure,
hold on,
grid minor,
for k=1:10
    
    plot(om,20*log10(abs(H(k,:)))) ;
    
    
end
plot(om,20*log10(absX(1,1:nfft/2)),'-r'),
title('dB of the filter function and magX');

clear y k;
y=x;

for k=1:10
    
    
    y=filter(b(k,:),a(k,:),y); % here we are filtering x;
    %k;
end


Y=fft(y,nfft);
magY=abs(Y(1:nfft/2));
figure,

plot(f,magY),
title('magY vs f');
figure,
plot(f,20*log10(magY)),title('dB of the magnitude of Y');




zoom xon;



%% b
close all;clc;
yy=y';
xx=x'; % I had to do this because the frames function 
% only accepts matrices nx1;
[xframe,xHoptime]=frames(xx,td,0.6,fs);
[yframe,yHoptime]=frames(yy,td,0.6,fs);
XX=fft(xframe,nfft);
YY=fft(yframe,nfft);
magXX=abs(XX(1:nfft/2+1,:));
magYY=abs(YY(1:nfft/2+1,:));
lmag=length(magXX(1,:));
ff=[0:lmag/2]*fs/lmag;

figure,

for k=1:lmag
%     subplot(2,1,1),plot(f(1:nfft/4),magXX(1:nfft/4,k),'-b'),title('Magnitude of X');
%     subplot(2,1,2),plot(f(1:nfft/4),magYY(1:nfft/4,k),'-r'),title('Maginitude of Y');
%     pause(0.05);
    subplot(2,1,1),plot(ff(1,k),magXX(:,k),'-b'),title('Magnitude of X');
    subplot(2,1,2),plot(ff(1,k),magYY(:,k),'-r'),title('Maginitude of Y');
    pause(0.05);
end



%zoom xon;
%% C. 
fh
Z=fhb*Y;


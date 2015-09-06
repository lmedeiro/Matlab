% Assignment 3
% Luiz Medeiros;

%% 1 Exercises with known Functions;
clear,close all,clc;
fs=200; % our sampling frequency Fs=200Hz;
f0=10; % our fundamental frequency from the function x is 10Hz;
t=0:1/fs:1; % the respective time, calculated with a period 1/fs;
w=2*pi*f0;
x=cos(w*t)+cos(2*w*t)+cos(4*w*t)+cos(8*w*t);
nfft=1024;
f=(0:nfft/2-1)*fs/nfft;
om=2*pi*f/fs;
fx=fft(x,nfft);
magFx=abs(fx(1:nfft/2));
subplot(2,1,1),plot(t,x),title('x[n] vs t'),xlabel('t'),ylabel('x[n]'),
subplot(2,1,2),plot(om,magFx),title('magnitude of the FT of x[n]'),ylabel('magFx'),
xlabel('omega');
zoom xon;

% b
N=3;
[C,L]=wavedec(x,N,'db4');
% LL=fft(C(1:201),nfft);
% magLL=abs(LL(1:nfft/2));
% figure,
% plot(om,magLL);
% length(LL)
beginPoint=1;
endPoint=1;
begin=1;
figure;
FY=[];
sr=fs;
for k=1:N+1
    endPoint=beginPoint+L(k)-1;
    wc(k)={C(beginPoint:endPoint)};
    d=cell2mat(wc(k));
    fd=fft(d,nfft);
    afd=abs(fd(1:nfft/2+1));
    FY=[FY afd]; % This is an improvised way of saving The result into a variable
    % to later graph the overall reconstruction result of the signal. 
    % time(k)={[0:
    beginPoint=endPoint+1;
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
nfft2=length(FY);
f2=(0:nfft2-1)*fs/nfft2;
%om2=pi*f/fs;
om2=linspace(0,pi,length(FY))

%length(f)
figure,
plot(om2,FY);
fy=fft(FY);
length(fy)
t2=linspace(0,1,length(fy));
figure,plot(t2,fy);
a=find(FY(1,:)>32);
b=zeros(1,length(FY));

for k=1:length(a)
    b(1,a(1,k))=FY(1,a(1,k));
    
end
figure,
plot(om2,b);
figure,plot(om2,b);

%% c
t=0:1/fs:1; % the respective time, calculated with a period 1/fs;
xRec=waverec(C,L,'db4');
figure,plot(t,xRec);
% By comparing the current plot with the one in figure 1 we are able to 
% see that the signals match and the algorithm does a very good job in 
% reconstructing the signal. 

%% problem 2
clear,close all,clc;
nfft=256;
fs=200;
[LO_D,HI_D,LO_R,HI_R]=wfilters('db4');
L=length(LO_D);
h=[LO_D',HI_D',LO_R',HI_R'];
H=fft(h,nfft);
absH=abs(H(1:nfft/2+1,:));

f=[0:nfft/2]/nfft*fs;

figure,
subplot(2,1,1),hold,
subplot(2,1,2),hold,
col='brgm';
for k=1:4
    subplot(2,1,1), plot(h(:,k), col(k)),title('DB Name= Db4'),xlabel('samples'),ylabel('filter values')
    subplot(2,1,2), plot(f,absH(:,k),col(k)),xlabel('f Hz'),ylabel('absH')
end
subplot(2,1,1), legend('LO_D', 'HI_D', 'LO_R', 'HI_R'),
subplot(2,1,2), legend('LO_D', 'HI_D', 'LO_R', 'HI_R');
%subplot(2,1,2),plot(f,absH(:,:));
% figure,
% subplot(2,1,1), plot(h(:,2), col(2)),title('DB Name= Db4'),xlabel('samples'),ylabel('filter values')
% subplot(2,1,2), plot(f,absH(:,2),col(2)),xlabel('f Hz'),ylabel('absH')

%% exercise 3
% e

clear; clc; close all;
%load scfn and wlt for db8.
load scfnL3db8    %scfnL3db3 % here we are loading the stored variables;
load wltL3db8    %wltL3db3     
M=length(wlt);  % here we are extracting the length of the signal or number of samples
%let the wavelet and scfn be of 50 ms duration
sr=M/50*10^3;   % Fs= sampling frequency= 1/T, where T=sampling period=50ms. sr=Fs;
%generate a toy signal from linear combinations of the scn
n=20;
cf=randn(1,n); % Here we are creating a random number array of 20. 
t=[0:1/sr:4]'; % by utilizing the given sampling rate sr, we split  a total time 
% of 4 seconds into our sampling period, 1/sr;
lx=length(t);
tr=[1:n]*M;
x=zeros(lx,1); % create an x array of zeros as big as our respective sample # 
for k=1:n
    x(tr(k):tr(k)+M-1)=x(tr(k):tr(k)+M-1)+cf(k)*scfn;   % Creating a signal based on randomizing the 
    % wavelets functions achieved from the previous exercise. 
end

figure
plot(t,x),title('Random Wavelet based signal'),xlabel('t,time'),ylabel('x');
dbname='db8';  % Daubechis 8 
nfft=1024; % Defining an nfft to perform the fft . 
fx=fft(x,nfft);% performing the fft 
afx=abs(fx(1:nfft/2+1));  % achieving absolute value of the fft ; 
f=(0:nfft/2)/nfft*sr; % defining the discrete frequency domain based on the nfft point DFT. 
fgn=1 % figure iteration number. 
figure(fgn)
subplot(211),plot(t,x), title ('random sig x vs t '),
subplot(212),plot(f,afx),title('abs value of the fft of x'),xlabel('f'),ylabel('abs(X)');
N=3; % filter level to decimate waves. 
[C,L]=wavedec(x,N,dbname); % Decimation of waves. returning values and coefficients to variables C and L
begin=1;
fgn=fgn+1; % incrementing the figure counter
figure(fgn)
str='';

for k=1:N+1
fin=begin+L(k)-1;
wc(k)={C(begin:fin)}; % Allocating the respective decimated signal to separate cells 
d=cell2mat(wc(k)); %placing the cells in one variable. 
fd=fft(d,nfft); % taking the fft of the cell with the decimated signals. 
afd=abs(fd(1:nfft/2+1)); % Now the absolute value of the function. 
begin=fin+1;
if k<2
    p=N % in the case we are going through our first iteration , we set p to level 3
else
    p=p-1;
end
t=(0:L(k)-1)/(sr/2^p); % creating the appropriate time domain. 
f=(0:nfft/2)/nfft*sr/2^p; %frequency representation. 
figure(fgn)

str=num2str(k);
subplot(N+1,2,2*k-1),plot(t,d),title(['decimated signal x ',str]) % ploting the raw value of the decimated portion of the function 
subplot(N+1,2,2*k),plot(f,afd),title(['decimated abs of the fft of x ',str]) % plotting the abs value of the same.  
end 
wx=ndwt(x,N,dbname,'mode','per'); % a struct with all of the pertaining information
% of a non dcimated transform.This includes: 
% level , N, signal , x, decimation algorithm name, dbname, mode, and
t=0:1/sr:4;
f=(0:nfft/2)/nfft*sr;
%a=indwt(wx,'a',0);
d4=indwt(wx,'d',4); % inverse the decimation by level 
d3=indwt(wx,'d',3);
d2=indwt(wx,'d',2);
d1=indwt(wx,'d',1);
 unc=[d1 d2 d3 d4];
 xx=sum(unc,2); % summing the inverse of the decimated signal. 
 recer=x-xx; % checking the difference between original signal and attained inversely decimate signal.
 figure
 plot(xx), title('inversely decimated x')
 hold
 pause
 plot(x,'r'), title('original x signal'),
 plot(recer,'g','LineWidth',2),title('difference between the two different xs');
 func=fft(unc,nfft); % going through the fft algorithm of the inverse operation. 
 afunc=abs(func(1:nfft/2+1,:));
 figure
 for k=1:N+1
     subplot(N+1,2,2*k-1), plot(t,unc(:,k))
     subplot(N+1,2,2*k),plot(f,afunc(:,k))
 end 
 %comparisons
 d=cell2mat(wc(1));
 figure
 subplot(211),stem(d(1:600))
 subplot(212), stem(tr/8+37,cf) % stem plot of the signal. 
 figure
 plot(t,x),title( 'original signal'),
 hold
 pause,figure, % I added the figure command because it is easier to just add it 
 % as another figure to view next to the original signal. 
 plot(t,unc(:,N+1),'r') % plot of the signal after gone through decimation and inverse decimation. 

%% 4. 
% Medeiros: db1,db6;

clear; clc; close all;
%load scfn and wlt for db8.
load scfnL3db6    %scfnL3db1 % here we are loading the stored variables;
load wltL3db6    %wltL3db1  % These were previously calculated and stored db1 waves    
load EEG4wlt;
M=length(wlt);  % 
sr=256;   % Fs= sampling frequency= 1/T, where T=sampling period=50ms. sr=Fs;
% This is more than sufficient to sample the given signal. 

t=[1/sr:1/sr:(length(x)/sr)]'; % by utilizing the given sampling rate sr, we split  a total time 
% time=sapmles/sampling rate=3600 second=60 minutes= 1hr. 
lx=length(t);

figure
plot(t,x),title('EEG Signal'),xlabel('t,time'),ylabel('x');
dbname='db6';  % Daubechis 8 
nfft=1024; % Defining an nfft to perform the fft . 
fx=fft(x,nfft);% performing the fft 
afx=abs(fx(1:nfft/2+1));  % achieving absolute value of the fft ; 
f=(0:nfft/2)/nfft*sr; % defining the discrete frequency domain based on the nfft point DFT. 
fgn=1 % figure iteration number. 
figure(fgn)
subplot(211),plot(t,x), title ('EEG sig x vs t '),
subplot(212),plot(f,afx),title('abs value of the fft of x'),xlabel('f'),ylabel('abs(X)');
N=3; % filter level to decimate waves. 
[C,L]=wavedec(x,N,dbname); % Decimation of waves. returning values and coefficients to variables C and L
begin=1;
fgn=fgn+1; % incrementing the figure counter
figure(fgn)
str='';
for a=1:3
for k=1:N+1
fin=begin+L(k)-1;
wc(a*k,:)={C(begin:fin)}; % Allocating the respective decimated signal to separate cells 
d=cell2mat(wc(a*k)); %placing the cells in one variable. 
fd=fft(d,nfft); % taking the fft of the cell with the decimated signals. 
afd=abs(fd(1:nfft/2+1)); % Now the absolute value of the function. 
begin=fin+1;
if k<2
    p=N % in the case we are going through our first iteration , we set p to level 3
else
    p=p-1;
end
t=(0:L(k)-1)/(sr/2^p); % creating the appropriate time domain. 
f=(0:nfft/2)/nfft*sr/2^p; %frequency representation. 
figure(fgn)

str=num2str(k);
subplot(N+1,2,2*k-1),plot(t,d),title(['decimated signal x ',str,'N is ',num2str(N)]) % ploting the raw value of the decimated portion of the function 
subplot(N+1,2,2*k),plot(f,afd),title(['decimated abs of the fft of x ',str,'N is ',num2str(N)]) % plotting the abs value of the same.  

end 
fgn=fgn+1;
N=N+1;
end


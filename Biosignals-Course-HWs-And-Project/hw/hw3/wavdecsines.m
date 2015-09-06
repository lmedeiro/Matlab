clear; clc; close all;
sr=200;
fo=10;
f1=20;
f2=40;
f3=80; 
t=0:1/sr:4;
x=cos(2*pi*fo*t)+cos(2*pi*f1*t)+cos(2*pi*f2*t)+cos(2*pi*f3*t);
lx=length(x);
nfft=1024;
fx=fft(x,nfft);
afx=abs(fx(1:nfft/2+1));
f=(0:nfft/2)/nfft*sr;
fgn=1
figure(fgn)
subplot(211),plot(t,x)
subplot(212),plot(f,afx)
dbname='db6';
N=3;
[C,L]=wavedec(x,N,dbname);
begin=1;
fgn=fgn+1;
figure(fgn)
for k=1:N+1
fin=begin+L(k)-1;
wc(k)={C(begin:fin)};
d=cell2mat(wc(k));
fd=fft(d,nfft);
afd=abs(fd(1:nfft/2+1));
begin=fin+1;
if k<2
    p=N
else
    p=p-1;
end
t=(0:L(k)-1)/(sr/2^p);
f=(0:nfft/2)/nfft*sr/2^p;
figure(fgn)
subplot(N+1,2,2*k-1),plot(t,d)
subplot(N+1,2,2*k),plot(f,afd)
end


wx=ndwt(x,N,dbname,'mode','per');
t=0:1/sr:4;
f=(0:nfft/2)/nfft*sr;
%a=indwt(wx,'a',0);

d4=indwt(wx,'d',4);
d3=indwt(wx,'d',3);
d2=indwt(wx,'d',2);
d1=indwt(wx,'d',1);
 unc=[d1' d2' d3' d4'];
 xx=sum(unc,2)';
 recer=x-xx;
 figure
 plot(xx)
 hold
 pause
 plot(x,'r')
 plot(recer,'g','LineWidth',2)
 func=fft(unc,nfft);
 afunc=abs(func(1:nfft/2+1,:));
 figure
 for k=1:N+1
     subplot(N+1,2,2*k-1), plot(t,unc(:,k))
     subplot(N+1,2,2*k),plot(f,afunc(:,k))
 end
 

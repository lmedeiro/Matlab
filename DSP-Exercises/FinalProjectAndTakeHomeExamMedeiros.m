

%% Final Project and Take Home Exam
%% Luiz Medeiros

%%%%%%%%%%%%%%%
%% I
%%%%%%%%%%%%%%%
clear, close all, clc;
%% A + B 
[x,sr,b]=wavread('TONE');
lx=length(x);   % Length of x[n];
t=[0:lx-1]/sr;  % Time in Seconds;
figure,
subplot(2,1,1),
plot(x),xlabel('samples');
subplot(2,1,2),
plot(t,x),xlabel('Time in seconds');

%% C + D; 

%nfft=2^(fix(log(lx)/log(2))+4);  % makes it log 2;
nfft=2^(fix(log2(lx))+4);   %  Utilizes log2(x); 
disp(nfft)
X=fft(x,nfft);
lX=length(X);
magX=abs(X(1:lX/2));
phaseX=angle(X(1:lX/2));
f=[0:lX/2-1]*sr/lX; % f in Hz. fs/2 -> pi; 
om=2*pi*f/sr; % omega;  2* pi * f;    

figure,
grid minor,
subplot(2,1,1),
plot(om,20*log10(magX)),
title('dB Plot of magX'),
subplot(2,1,2),
plot(om,magX),
title('magX vs Omega; 0-> pi');

%% E
figure,
grid minor,
subplot(2,1,1),
plot(f,20*log10(magX)),
title('dB Plot of magX'),
subplot(2,1,2),
plot(f,magX),
title('magX vs f; 0-> pi');

%% F
% Using the Spectral tip, find the four highest peaks;
% f0=1330;f1=3989.94;f2=6649.93;f3=9309.88;


%% G

w1=1330/sr*2*pi;
w2=3989.94/sr*2*pi;
w3=6649.93/sr*2*pi;
w4=9309.88/sr*2*pi;
%w5=11969.84/sr*2*pi;
w=[w1 w2 w3 w4];

%% H
clear w0 H r a b k;
H=[zeros(5,nfft/2)];
r=0.99;
w0=[zeros(1,4)];
a=[zeros(4,3)];
b=[zeros(4,3)];
% Part 4
for k=1:4
    w0=w(1,k);
    b(k,:)=[1 -2*cos(w0) 1];
    a(k,:)=[1 -2*r*cos(w0) r^2];
    H(k,:)=freqz(b(k,:),a(k,:),nfft/2);
end
%
%% H part 5
figure,
hold on,
grid minor,
plot(f,20*log10(abs(H)));
 
title('dB of the filters and magX');

%% I
clear y k;
y=x;

for k=1:4
    
    
    y=filter(b(k,:),a(k,:),y);
    %k;
end
%% I part 1
% Paper;

%% I part 2
clear hImpf hImpt;
hImpf=zeros(4,length(f));
hImpt=zeros(4,length(t));

for k=1:4
    
    hImpf(k,:)=filter(b(k,:),a(k,:),[1,zeros(1,length(f)-1)]);
    hImpt(k,:)=filter(b(k,:),a(k,:),[1,zeros(1,length(t)-1)]);

    
end

figure,
hold on, grid minor,
title('hImpf vs f'),
for k=1:4
    
    subplot(4,1,k),
    plot(f,hImpf(k,:)),
    title('hImpf vs f'),
    axis([-1 100 0 1.1]);
    
end
figure,
plot(f,hImpf(1,:)),axis([-1 100 0 1.1]);
figure,plot(t,hImpt(1,:));

%% J part 1;
% The combining operation we must use to obtain H(z), 
% with all the frequencies we wish to filter is the convolution
% of all coefficients of b and the convolution of all coefficients
% of a. This will result in the b/a form that after the 
% performing the z transform gives us our H(z).

ha=a(1,:);
hb=b(1,:);
for k=2:4
    ha=conv(ha,a(k,:));
    hb=conv(hb,b(k,:));
end
HH=freqz(hb,ha,nfft/2);

%% J part 2; 

% The lengths N and M are respectively 9. 

%% J part 3;
% The coefficients are as follows: 
bn=hb
an=ha

%% J part 4; 
s=filter(hb,ha,x);
S=fft(s,nfft);
magS=abs(S(1:nfft/2));


%% J part 4 a; 
Y=fft(y,nfft);
magY=abs(Y(1:nfft/2));
figure,
subplot(2,1,1),
plot(f,magY),
title('magY vs f');
subplot(2,1,2),
plot(f,magS),
title('magS vs f');


figure,
subplot(2,1,1),
plot(f,20*log10(magY)),
title('dB of magY vs f');
subplot(2,1,2),
plot(f,20*log10(magS)),
title('dB of magS vs f');

figure,
subplot(2,1,1),
plot(t,s),title('s vs t');
subplot(2,1,2),plot(t,y),
title('y vs t');

% Graphically, I am not able to see any difference. 
% I zoomed and analyzed the both images and graphs, however,
% no apparent difference was found. 

%% J part 4 b;
soundsc(y,sr); 
soundsc(s,sr);

% Within both hearings I hear a significant decrease in noise. 
% The loud beep that followed the voice in the beginning is no longer
% present. However there is still some noise present, but I am 
% not able to distinguish the noise or sound difference between
% either output. 

%% J part 4 c;

d=s-y;

%% J part 4 c i; 

figure,
subplot(2,1,1),
plot(t,s),title('s vs t');
subplot(2,1,2),plot(t,y),
title('y vs t');
figure,plot(t,d),title('d vs t'); 

% From the graph and vector entries, it is possible to see magnitudes 
% which range from -0.0049 to 0.0047. A small, but yet
% reasonable error. 
% From what was mentioned in class, this reapproves the point 
% where the latter S(z) contains less error than Y(z). 


%% J part 4 c ii;
D=fft(d,nfft);
magD=abs(D(1:nfft/2));
energy= (magD./magY).^2;
plot(om,energy),
title('energy vs omega');
[q w]=max(energy);
w*2*pi/lX;

% From analyzing the graph given, it is possible to see an 
% energy spike in each frequency that we ommitted from the 
% original signal. Very small (10^(-14)), but still something 
% to consider. 






%% Tests
% 
% Y=fft(y,nfft);
% magY=abs(Y(1:nfft/2));
% figure,
% plot(f,magY),
% title('magY vs f');
% 
% figure,
% plot(f,20*log10(magY)),
% title('dB of magY vs f');

% h1bh2b=conv(b(1,:),b(2,:))
% hb=conv(h1bh2b,b(3,:))
% ha=a(1,:);
% hb=b(1,:);
% for k=2:4
%     ha=conv(ha,a(k,:));
%     hb=conv(hb,b(k,:));
% end



% f=[0:length(magX)-1]/length(magX); % f in Hz. 
% om=2*pi*sr*f;
% figure,
% plot(magX);
%figure,
%plot(om,magX);

% b= [1 -2*cos(w0) 1];
% r= 0.99;
% a= [1 -2*r*cos(w0) r^2]; % Coefficient of the denominator. 
% H0= freqz(b,a,N/2);
% 
% b= [1 -2*cos(w1) 1];
% r= 0.99;
% a= [1 -2*r*cos(w1) r^2]; % Coefficient of the denominator. 
% H1= freqz(b,a,N/2);
% 
% b= [1 -2*cos(w2) 1];
% r= 0.99;
% a= [1 -2*r*cos(w2) r^2]; % Coefficient of the denominator. 
% H2= freqz(b,a,N/2);
% 
% b= [1 -2*cos(w3) 1];
% r= 0.99;
% a= [1 -2*r*cos(w3) r^2]; % Coefficient of the denominator. 
% H3= freqz(b,a,N/2);
% 
% b= [1 -2*cos(w4) 1];
% r= 0.99;
% a= [1 -2*r*cos(w4) r^2]; % Coefficient of the denominator. 
% H4= freqz(b,a,N/2);














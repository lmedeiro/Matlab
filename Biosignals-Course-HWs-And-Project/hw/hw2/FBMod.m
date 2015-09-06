N=30;
h=firpm(N,linspace(0,1,30),[1 1 1 0.707 zeros(1,26)]);
% fvtool(h)
%cosine modulated FBs
hb=zeros(N+1,6);
hb(:,1)=h';
for k=1:5
    hb(:,k+1)=h'.*cos(0.2*pi*k*[0:N]')*2;
end
hb(:,6)=h'.*cos(0.2*pi*k*[0:N]');
 fhb=fft(hb,256);
afhb=abs(fhb(1:129,:));
f=[0:128]'/128;
plot(f,20*log10(afhb))
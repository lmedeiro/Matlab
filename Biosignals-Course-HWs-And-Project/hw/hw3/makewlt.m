clear; %clc; 
close all;
sr=200;
dbname='db6';
[lod, hid, lor, hir]=wfilters(dbname); % Takes the filter coefficients
L=length(lod);
P=8
K=2^(P+1)-1;
%total filter length
M=K*L-K+1-P;
nfft=2^12;
nfft=max(2^(fix(log2(M))+2),nfft);
h=[lod' hid' lor' hir'];
fh=fft(h,nfft);
afh=abs(fh(1:nfft/2+1,:));
f=[0:nfft/2]/nfft*sr;
figure
subplot(2,1,1), hold
subplot(2,1,2), hold
col='brgm';
for k=1:4
    subplot(2,1,1), plot(h(:,k), col(k))
    subplot(212), plot(f,afh(:,k),col(k))
end
subplot(2,1,1), legend('lod', 'hid', 'lor', 'hir')
subplot(2,1,2), legend('lod', 'hid', 'lor', 'hir')
%Implement G( 2^4 w) H(2^3 w)H(2^2 w) H(w) 


% Book mark point on the project

% restart here; 
fscfn=repmat(downsample(fh(:,1),2^P),2^P,1); % (low filter) replicates the first column of fh 2^P times in a row fashion.  
fwlt=repmat(downsample(fh(:,2),2^P),2^P,1);  % (high filter)

for k=0:P-1
    
    b=repmat(downsample(fh(:,1),2^k),2^k,1);
    fscfn=fscfn.*b;
    fwlt=fwlt.*b;
end

scfn=ifft(fscfn);
scfn=real(scfn); 
wlt=ifft(fwlt);
wlt=real(wlt);
figure
subplot(211),plot(scfn),grid minor
subplot(212),plot(wlt), grid minor
wlt=wlt(1:M);
scfn=scfn(1:M);
figure
subplot(211),plot(scfn),grid minor
subplot(212),plot(wlt), grid minor
save wltL3db6 wlt 
save scfnL3db6 scfn 


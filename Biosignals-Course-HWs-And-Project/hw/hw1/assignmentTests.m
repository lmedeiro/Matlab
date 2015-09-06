% assignment 1 tests
close all; clear; clc; % just doing some house keeping;
load('edbe0103'); % loading the file;
fc=sig(:,1);    % setting my own variable to the signal captured in channel 1;
% fc= first channel;
I=find(fc>1); % using the find function from matlab to help us find all of 
% the item that are above 1, where 1 is our treshold;
%x=zeros(length(I));
x=ones(1800000,1); % here we initiate the first dummy var x, which will 
% contain all the values on their respective indexes coming from I;
n=1; % our main iterative variable
while n<=61844
    x(I(n))=fc(I(n)); % here we have a the index and value setting. 
    n=n+1;
end
%figure,plot(fc),hold,plot(x,'r*'); % this was mainly for test purposes. 
n=1; % resetting our iterative variable to keep some consistency. 
mx=[];% denotes the maxima; first row is the spot, second is the value. 
s=1;v=2; % (s)pot, (v)alue;
j=1;    % our second iterative variable used; 
% this loop basically sets up the maxima array so we have maxima and
% positions.
while n<(length(x)-1)
    diff=x(n+1)/x(n)-1; % this is the logic we follow to find the where
    % the maxima is located. Basically where there occured a change of 
    % signs from positive to negative.  
    
    if diff<0
       mx(s,j)=n;
       mx(v,j)=x(n);
       j=j+1;
       %n=n+1;
       %diff=x(n+1)/x(n)-1;
       
       % below is the iterative logic to get to the next positive position.
       while diff<=0 && n<(length(x)-1)
           diff=x(n+1)/x(n)-1;
           n=n+1;
       end
    else
        n=n+1;
    end
end
n=1;
% we now create another dummy matrix to store only the maxima. 
% we then plot it against the original first channel samples. 
y=ones(1800000,1);
while n<length(mx)
    y(mx(s,n))=mx(v,n);
    n=n+1;
    
end

plot(tm,fc),hold,plot(tm,y,'r*'),xlabel('Time t (tm)'),
ylabel('Recorded Sample values'),title('Recorded Samples vs Time t'); % our maxima are denoted as red stars. 


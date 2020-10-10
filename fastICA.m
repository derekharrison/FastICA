% An implementation of the fast ICA algorithm

clear
clc

tic

%% Parameters

N           = 6;                            %The number of observed mixtures
C           = N;                            %The number of independent sources
M           = 1000;                         %Sample size, i.e.: number of observations

K           = 0.1;                          %Slope of zigzag function
na          = 8;                            %Number of zigzag oscillations within sample
ns          = 5;                            %Number of alternating step function oscillations within sample

finalTime   = 40*pi;                        %Final sample time (s)
initialTime = 0;                            %Initial sample time (s)

%% Generating Data for ICA

Amix        = rand(N,N);                    %Amix is a random N x N mixing matrix

timeVector  = initialTime:(finalTime-initialTime)/(M-1):finalTime;  %Vector of time coordinates

source1     = sin(1.1*timeVector);          %Independent source component 1, sin(a * t)
source2     = cos(0.25*timeVector);         %Independent source component 2, cos(b * t)
source3     = sin(0.1*timeVector);          %Independent source component 3, sin(c * t)
source4     = cos(0.7*timeVector);          %Independent source component 4, cos(d * t)
source5     = zeros(1,M);                   %Independent source component 5, a zigzag pattern
source6     = zeros(1,M);                   %Independent Component 6, alternating step-function

periodSource5 = (finalTime-initialTime)/na;
periodSource6 = (finalTime-initialTime)/ns/2;

for i = 1:M
    source5(i) = K*timeVector(i)-floor(timeVector(i)/periodSource5)*K*periodSource5;
end

source5 = source5 - mean(source5);

for i = 1:M
    if mod(floor(timeVector(i)/periodSource6),2) == 0
        source6(i) = 1;
    else
        source6(i) = -1;
    end    
end

source6 = source6 - mean(source6);

S = [source1;source2;source3;source4;source5;source6];    %Source Matrix

Xobs = Amix*S;                              %Matrix consisting of M samples of N observed mixtures

figure
plot(timeVector,source1)                    %Plotting the N independent sources vs. time
xlabel('time (s)')
ylabel('Signal Amplitude') 
legend('source 1')

figure
plot(timeVector,source2)                    %Plotting the N independent sources vs. time
xlabel('time (s)')
ylabel('Signal Amplitude') 
legend('source 2')

figure
plot(timeVector,source3)                    %Plotting the N independent sources vs. time
xlabel('time (s)')
ylabel('Signal Amplitude') 
legend('source 3')

figure
plot(timeVector,source4)                    %Plotting the N independent sources vs. time
xlabel('time (s)')
ylabel('Signal Amplitude') 
legend('source 4')

figure
plot(timeVector,source5)                    %Plotting the N independent sources vs. time
xlabel('time (s)')
ylabel('Signal Amplitude') 
legend('source 5')

figure
plot(timeVector,source6)                    %Plotting the N independent sources vs. time
xlabel('time (s)')
ylabel('Signal Amplitude') 
legend('source 6')

figure
plot(timeVector,Xobs);                      %Plotting N observed mixtures of each sample vs. time
xlabel('time (s)') 
ylabel('Signal Amplitude')
legend('Observed Mixture 1', 'Observed Mixture 2', 'Observed Mixture 3', 'Observed Mixture 4', 'Observed Mixture 5', 'Observed Mixture 6')

%% Preprocessing, Centering

X = Xobs';                                  %X is the transpose of the matrix of M samples of N mixtures, used in subsequent calculations

Xmean = mean(X);                            %Xmean is the mean vector of the matrix X

for i = 1:N
    X(:,i) = X(:,i) - Xmean(i);             %The matrix X is centered by subtracting each of the N mixtures by their corresponding sample averages
end

%% Preprocessing, Whitening

ExxT    = cov(X);                           %The covariance matrix of X is computed and stored in ExxT
[E,D]   = eig(ExxT);                        %Eigenvalue decomposition is applied on the covariance matrix of X, ExxT

Z = E*1/sqrt(D)*E'*X';                      %The matrix X is whitened to Z

%% FastICA algorithm

W = 0.5*ones(C,N);                              %Initializing W, a matrix consisting of columns corresponding with the inverse of the (transformed) mixing Amix

iterations = 100;                           %The amount of iterations used in the fastICA algorithm

for p = 1:C
    
    wp = ones(N,1)*0.5;
    wp = wp / sqrt(wp'*wp);

    for i = 1:iterations
        
        G       = tanh(wp'*Z);
        Gder    = 1-tanh(wp'*Z).^2;
        
        wp      = 1/M*Z*G' - 1/M*Gder*ones(M,1)*wp;
        
        dumsum  = zeros(C,1);
        
        for j = 1:p-1
            dumsum = dumsum + wp'*W(:,j)*W(:,j);
        end
        
        wp      = wp - dumsum;        
        wp      = wp / sqrt(wp'*wp);
    end
    
    W(:,p) = wp; 
end

%% Output Results

W = W/sqrt(2);                              %The factor sqrt(2) is an empirical constant added to make the predictions fit the data properly. The source of the factor has yet to be determined.

W;

Ainvest = W'*E*1/sqrt(D)*E'  ;               %The estimated inverse of the mixing matrix Amix. Note order of rows may vary. 
Ainvact = inv(Amix)        ;                 %The actual inverse of the mixing matrix Amix

Sest = W'*Z;

figure
plot(timeVector, Sest(1,:))
xlabel('time (s)') 
ylabel('Signal Amplitude') 
legend('Source Estimation 1')

figure
plot(timeVector, Sest(2,:))
xlabel('time (s)') 
ylabel('Signal Amplitude') 
legend('Source Estimation 2')

figure
plot(timeVector, Sest(3,:))
xlabel('time (s)')
ylabel('Signal Amplitude') 
legend('Source Estimation 3')

figure
plot(timeVector, Sest(4,:))
xlabel('time (s)')
ylabel('Signal Amplitude') 
legend('Source Estimation 4')

figure
plot(timeVector, Sest(5,:))
xlabel('time (s)')
ylabel('Signal Amplitude') 
legend('Source Estimation 5')

figure
plot(timeVector, Sest(6,:))
xlabel('time (s)')
ylabel('Signal Amplitude') 
legend('Source Estimation 6')

toc

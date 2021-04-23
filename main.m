% LE ARNING FROM EXPERIENCE IN THE STOCK MARKET
% GALO NUNO & ANTON NAKOV
% This version: July 2014

tic
clear; close all
clc
randn('state',0)

% PARAMETERS
data;                 % load Shiller's data 
mu = mean(dlogDd);    % mean monthly dividends growth rate from Shiller
sigma = std(dlogDd);  % std of monthly dividends growth rate from Shiller
avlife = 12*40;       % average life in months
phi = 1 - 1/(avlife); % survival probability per month
avPD = 12*mean(PDd);  % average P/D ratio, monthly
beta = avPD/((1+avPD)*phi*exp(mu+sigma^2/2)); % discount to hit avPD under REE
theta = 3.044/3;      % parameter in decreasing gain from Malmendier-Nagel
S = 2*avlife;         % total number of age cohorts 
N = 1;                % number of Monte Carlo Simulations
T = length(Dd);       % sample size 

% Calibration of the dotcom bubble
t1 = find(dates>=1995,1,'first'); % P/D boom beginning
t2 = find(dates>=2000.6,1,'first'); % P/D boom peak
PDlowceil = 12*40;    % P/D low ceiling
PDhiceil = 12*120;    % P/D high ceiling
lambda = NaN(T,1);    % maximum leverage (multiple of D)
lambda(:) = PDlowceil;   
  %lambda(1:t1) = PDlowceil;   
  %lambda(t1:t2) = PDlowceil:((PDhiceil-PDlowceil)/(t2-t1)):PDhiceil;   
  %lambda(t2:T) = PDhiceil;   
     %lambda(t2:T) = PDhiceil:-((PDhiceil-PDlowceil)/(T-t2)):PDlowceil;   



% ALLOCATE MEMORY FOR VARIABLES
D = NaN(1,T,N);           % dividends
P = NaN(1,T,N);           % prices
PD = NaN(1,T,N);          % PD ratio
PREE = NaN(1,T,N);        % prices REE
mD = NaN(S,T,N);          % belief for growth rate of dividends
mP = NaN(S,T,N);          % belief for growth rate of prices
mD_av = NaN(1,T,N);       % mean mD across cohorts
mP_av = NaN(1,T,N);       % mean mP across cohorts
dlogD = NaN(1,T,N);       % growth rate of dividends   
dlogP = NaN(1,T,N);       % growth rate of prices  
pos = NaN(1,T,N);         % position of the marginal agent
gamma = NaN(S,1,N);       % decreasing gain parameter in Malmendier-Nagel
ff = NaN(S,1,N);          % share of agents in each age cohort
errorD = NaN(S,T-1,N);    % forecasting error in dividends
errorP = NaN(S,T-1,N);    % forecasting error in prices
errorDREE = NaN(1,T-1,N); % forecasting error in dividends REE
errorPREE = NaN(1,T-1,N); % forecasting error in prices REE
res_prices = NaN(S,T,N);  % reservation prices

% INITIALIZATION (this shouldn't matter because cohort 2 ignores cohort 1)
mD(:,1,:) = mu;     % time t=1 belief is REE 
mP(:,1,:) = mu;     % time t=1 belief is REE 
mD(1,:,:) = mu;     % cohort s=1 belief is REE
mP(1,:,:) = mu;     % cohort s=1 belief is REE
% shock = sigma*randn(1,T,N); % shock to dividends

% REE PD ratio 
PDREE = (beta*phi*exp(mu))/(1-beta*phi*exp(mu)); 


% MONTE CARLO LOOP
for n = 1:N
    
% DIVIDEND PROCESS    
D(:,:,1) = Dd; % Take dividends from the data
dlogD(1,2:T,n) = log(D(1,2:T,n)./D(1,1:T-1,n));

% D(1,1,:) = 1;       % initial dividend
% for t=2:T
%    D(1,t,n) = D(1,t-1,n)*exp(mu + shock(1,t,n));
% end

gamma(1:floor(theta),1,:) = 1;
gamma((floor(theta)+1):S,1,n) = theta./((floor(theta)+1):S);

ff(:,1,n)   = (1-phi) * phi.^(0:S-1);

% REE SOLUTION for the price
PREE(1,:,n)  = PDREE .* D(1,:,n);



% Initial price
P(1,1,n) = Pd(1);     
% P(1,1,n) = PDREE*D(1,1,n);     

mD_av(1,1,n) = sum(mD(:,1,n) .* ff(:,1,n))/sum(ff(:,1,n));
mP_av(1,1,n) = sum(mP(:,1,n) .* ff(:,1,n))/sum(ff(:,1,n));


for t=2:T
           
    % Updating of dividends growth
    mD(2:S,t,n) = mD(1:S-1,t-1,n) + gamma(2:S,1,n) .* (dlogD(1,t,n) - mD(1:S-1,t-1,n)); 
        
    % Market clearing price
    P(1,t,n) = fzero(@(Pguess) learningP(Pguess,D,P,mP,mD,gamma,S,n,t,lambda(t),beta,phi,ff,dlogP),P(1,t-1,n));
    
    dlogP(1,t,n) = log(P(1,t,n)./P(1,t-1,n));
    
    % Updating of price growth
    mP(2:S,t,n)  = mP(1:S-1,t-1,n) + gamma(2:S,1,n) .* (dlogP(1,t,n) - mP(1:S-1,t-1,n));
        
    % Reservation prices
    res_prices(:,t,n) = beta*phi*(exp(mP(:,t,n))* P(1,t,n) + exp(mD(:,t,n))* D(1,t,n)); 
    
    pos(1,t,n) = position(P(1,t,n),D,P,mP,mD,gamma,S,n,t,lambda(t),beta,phi,ff,dlogP);

    mD_av(1,t,n) = sum(mD(:,t,n) .* ff(:,1,n))/sum(ff(:,1,n));  % average value
    
    mP_av(1,t,n) = sum(mP(:,t,n) .* ff(:,1,n))/sum(ff(:,1,n)); % aver.value
    
    if rem(t,25)==2 || t==T
        clc
        fprintf('Simulation: %d of %d \n', [n, N])
        fprintf('Year: %1.1f \n\n', dates(t))
    end

end

end

toc

figs   

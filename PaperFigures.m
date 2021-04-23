%%---------------------------
%%TABLES
%-----------------------------
%TABLE1
disp('Table 1')
mean(mean(dlogP,3),2)
mean(mean(dlogD,3),2)

for i=1:N
     PD(1:T,i)        = P(1,1:T,i)./D(1,1:T,i);
     meanPD(i)        = mean(PD(:,i));
     stddlogP(i)      = std(dlogP(1,:,i));
     stddlogD(i)      = std(dlogD(1,:,i));
     stdPD(i)         =   std(dlogD(1,:,i));
end
mean(meanPD)
mean(stddlogP)
mean(stddlogD)
mean(stdPD)
-----------------------------
%TABLE2
for i=1:N
    mean_mP_CGL(i) = mean(mP_CGL(1,:,i));
    std_mP_CGL(i)  = std(mP_CGL(1,:,i));
    mean_xiP(i)    = mean(xiP(1,:,i));
    std_xiP(i)     = std(xiP(1,:,i));
    corrPCGL(i)    = corr(mP_av(1,:,i)',mP_CGL(1,:,i)');
    mean_mD_CGL(i) = mean(mD_CGL(1,:,i));
    std_mD_CGL(i)  = std(mD_CGL(1,:,i));
    mean_xiD(i)    = mean(xiD(1,:,i));
    std_xiD(i)     = std(xiD(1,:,i));
    corrDCGL(i)    = corr(mD_av(1,:,i)',mD_CGL(1,:,i)');
end
disp('Table 2')
    mean(mean_mP_CGL)
    mean(std_mP_CGL)
    mean(mean_xiP)
    mean(std_xiP)
    mean(corrPCGL)
    mean(mean_mD_CGL)
    mean(std_mD_CGL)
    mean(mean_xiD)
    mean(std_xiD)
    mean(corrDCGL)
%---------------------------------
% FIGURES
%----------------------------------
% %FIGURE 1: PLOTTING ORDERS
t=480;
figure
subplot(211)
[P_ord,index] = sort(beta*phi*(exp(mD(:,t,n))*D(1,t,n)+exp(mP(:,t,n))*P(1,t,n)),'descend');
plot(P_ord/PREE(1,t,n), 'Linewidth',2)
hold on
plot(P(1,t,n)/PREE(1,t,n)*ones(size(P_ord)),'r--', 'Linewidth',2)
axis 'tight'
legend('Reservation prices','Market price','Location','Best')
xlabel('Ordered cohorts')
ylabel('Asset Price / Price REE')
subplot(212)
plot(PDH * D(1,t,n) /P(1,t,n) *cumsum(ff(index,1,n))/sum(ff(:,1,n)), 'Linewidth',2);
hold on
plot(ones(size(P_ord)),'r--', 'Linewidth',2)
xlabel('Ordered cohorts')
ylabel('Cumulative asset demand (S)')
legend('Cumulative demand','Total amount','Location','Best')
axis 'tight'
%-----------------------------
%------------------
% % %FIGURES 2 AND 3 PRICE DIVIDEND RATIOS
figure
subplot(311)
plot(P(1,S:T,n)./PREE(1,S:T,n));
axis([1 570 0.9 1.2])
title('Asset price / Price REE')
subplot(312)
plot(exp(mP_av(1,S:T,n)).*P(1,S:T,n)./PREE(1,S:T,n),'Linewidth',1.5);
hold on
plot(exp(mP(2,S:T,n)).*P(1,S:T,n)./PREE(1,S:T,n),'r--');
plot(exp(mP(S,S:T,n)).*P(1,S:T,n)./PREE(1,S:T,n),'k:');
axis([1 570 0.8 1.3])
legend('Mean','Youngest cohort','Oldest cohort')
title('Expectations of price next period / Price REE')
subplot(313)
PD = P(1,S:T,n)./D(1,S:T,n);
plot(PD)
axis([1 570 8.0 10.5])
title('Price-dividend ratio')
xlabel('quarters')
%---------------------------------------
%FIGURES 4 & 5: FORECASTING ERRORS
%--------------------------------
figure
subplot(221)

for i=1:N
    errorP(1:S,:,i)      = ones(S,1)*dlogP(1,2:T,i) - mP(1:S,1:T-1,i);
    errorPREE(1,1:T-1,i) = log(PREE(1,2:T,i)./PREE(1,1:T-1,i)) - mu;
end

plot(mean(mean(errorP(:,:,:),2),3),'Linewidth',2)
hold on
plot(ones(S,1)*mean(mean(errorPREE(1,:,:)),3),'r:','Linewidth',2)
legend('HA-OLG','REE', 'Location','Best')
xlabel('age cohort')
title('Mean forecasting error - Prices')
% legend 'boxoff'
axis([0 S -10e-4 10e-4])

subplot(223)
std_errorP    = zeros(S,1,N);
std_errorPREE = zeros(1,1,N);
for i=1:N
    std_errorP(:,1,i)    =  std(errorP(:,:,i)');
    std_errorPREE(1,1,i) = std(errorPREE(1,:,i));
end

plot(mean(std_errorP(:,1,:),3),'Linewidth',2)
hold on
plot(ones(S,1)*mean(std_errorPREE(1,1,:),3),'r:','Linewidth',2)
% legend('HA-OLG','REE', 'Location','Best')
xlabel('age cohort')
title('Forecasting RMSE - Prices')
% legend 'boxoff'
axis([0 S 0.0 0.08])

subplot(222)
for i=1:N
    errorD(1:S,:,i)     = ones(S,1)*log(D(1,2:T,i)./D(1,1:T-1,i)) - mD(1:S,1:T-1,i);
    errorDREE(1,1:T-1,i) =log(D(1,2:T,i)./D(1,1:T-1,i)) - mu;
end

plot(mean(mean(errorD(:,:,:),2),3),'Linewidth',2)
hold on
plot(ones(S,1)*mean(mean(errorDREE(1,:,:)),3),'r:','Linewidth',2)
% legend('HA-OLG','REE', 'Location','Best')
xlabel('age cohort')
title('Mean forecasting error - Dividends')
% legend 'boxoff'
axis([0 S -10e-4 10e-4])

subplot(224)
std_errorD    = zeros(S,1,N);
std_errorDREE = zeros(1,1,N);
for i=1:N
    std_errorD(:,1,i)    =  std(errorD(:,:,i)');
    std_errorDREE(1,1,i) =  std(errorDREE(1,:,i));
end

plot(mean(std_errorD(:,1,:),3),'Linewidth',2)
hold on
plot(ones(S,1)*mean(std_errorDREE(1,1,:),3),'r:','Linewidth',2)
% legend('HA-OLG','REE', 'Location','Best')
xlabel('age cohort')
title('Forecasting RMSE - Dividends')
% legend 'boxoff'
axis([0 S 0.0 0.08])

%--------------------------------
%---------------------------------
% %FIGURE 6  CONSTANT GAIN PARAMETER
phi1 = (0.986:0.0005:0.999);
theta1 = (1:0.1:3);
gamma1 =zeros(length(phi1),length(theta1));

for i=1:length(phi1)
    for j=1:length(theta1)
        gamma1(i,j)= (1-phi1(i))*(-theta1(j)*log(1-phi1(i)));
        for s=1:floor(theta1(j))
           gamma1(i,j)=gamma1(i,j)+ (1-phi1(i))*phi1(i)^s *(1-theta1(j)/s);
        end
    end
end
surf(theta1,phi1,gamma1)         
axis 'tight'
xlabel('\phi')
xlabel('\theta')
ylabel('\phi')
zlabel('\gamma')

% %FIGURES 7-8 OPEN AND CLOSED LOOP APPROXIMATION
subplot(211)
plot(100*mP_CGL(1,T-200:T,n))
hold on
plot(100*mP_av(1,T-200:T,n),'r:')
title('Expectations of price growth m^P_t')
ylabel('%')
axis 'tight'
subplot(212)
plot(100*mD_CGL(1,T-200:T,n))
hold on
plot(100*mD_av(1,T-200:T,n),'r:')
axis 'tight'
title('Expectations of dividend growth m^D_t')
legend('RA-CGL','HA-OLG') 
ylabel('%')
% 


%FIGURE 9
figure
subplot(211)
n = 1;
line = 'b';
PD = P(1,S:T,n)'./D(1,S:T,n)';
plot(PD,'r:','Linewidth',2)
hold on
PD_CGL = P_CGL(1,S:T,n)'./D(1,S:T,n)';
plot(PD_CGL,line)
legend('HA-OLG','RA-CGL, \theta = 3.044') 
title('Price-dividend ratio')
xlabel('quarters')
axis([1 570 6.0 13])

subplot(212)
n = 1;
line = 'b';
PD = P(1,S:T,n)'./D(1,S:T,n)';
plot(PD,'r:','Linewidth',2)
hold on
PD_CGL = P_CGL(1,S:T,n)'./D(1,S:T,n)';
plot(PD_CGL,line)
legend('HA-OLG','RA-CGL, \theta = 1') 
title('Price-dividend ratio')
xlabel('quarters')
axis([1 570 6.0 13])

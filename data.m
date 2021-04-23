%% Computes moments from Shiller's dataset

dataseries = xlsread('ie_data.xls','Data');
Pd = dataseries(1:end-1,8);
Dd = dataseries(1:end-1,9);

dlogPd = log(Pd(2:end)./Pd(1:end-1));
dlogDd = log(Dd(2:end)./Dd(1:end-1));
PDd = Pd./Dd;

dates = (1871+1/24):(1/12):2014.47;

disp('[1200*mean(dlogP) 1200*std(dlogP) 1200*mean(dlogD) 1200*std(dlogD) mean(PD) std(PD)]')    
[1200*mean(dlogPd) 1200*std(dlogPd) 1200*mean(dlogDd) 1200*std(dlogDd) mean(PDd) std(PDd)]'    

%% Plots figures 1 and 2
if 0


figure(1)

subplot(211)
plot(dates,Dd)
title('Dividend')
ylim([0 40])
xlim([dates(1) dates(end)])

subplot(212)
plot(dates,Pd)
title('Price')
ylim([0 2500])
xlim([dates(1) dates(end)])

figure(2)
plot(dates,PDd)
legend('Price-dividend ratio','Location','NorthEast')
legend boxoff
ylim([0 100])
xlim([dates(1) dates(end)])

end
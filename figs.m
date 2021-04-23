s = 2;

%% TABLE1

PD = NaN(1,T,N);         % PD ratio
for i=1:N
     meandlogP(i) = mean(dlogP(1,s:T,i));
     meandlogD(i) = mean(dlogD(1,s:T,i));
     stddlogP(i) = std(dlogP(1,s:T,i));
     stddlogD(i) = std(dlogD(1,s:T,i));
     PD(1,:,i) = P(1,:,i)./D(1,:,i);
     meanPD(i) = mean(PD(1,s:T,i));
     stdPD(i) = std(PD(1,s:T,i));
end
disp('Table 1')
[1200*mean(meandlogP) 1200*mean(stddlogP) 1200*mean(meandlogD) 1200*mean(stddlogD) mean(meanPD)/12 mean(stdPD)/12]'


%% FIGURES 3 and 4

figure(3)

subplot(211)
plot(dates,D)
title('Dividend')
ylim([0 40])
xlim([dates(1) dates(end)])

subplot(212)
plot(dates,P)
title('Price')
%ylim([0 2500])
xlim([dates(1) dates(end)])

figure(4)
plot(dates,PD/12)
hold on
plot(dates,PDd,'r--')
plot(dates,PDREE/12*ones(size(dates)),'k-.','LineWidth',1.5)
title('Price-dividend ratio')
legend('Model','Data','REE','Location','NorthEast')
legend boxoff
ylim([0 100])
%xlim([dates(1) dates(end)])
xlim([1920 2014.5])

figure(5)
hold on
for s=[2 5 10 25 50 100 300 500  S]
plot(dates,mP(s,:,n),'Color',[0 (S-s)/S (S-s)/S]);
end
%axis([1 570 0.8 1.3])
title('Expectations of price growth: darker color indicates older cohort')
xlim([1920 2014.5])



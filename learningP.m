function Z=learningP(Pguess,D,P,mP,mD,gamma,S,n,t,lambda,beta,phi,ff,dlogP)
% Employed to compute the fixed-point price that clears the market
    
% CASE OF POSITION CONSTRAINTS
dlogP(1,t,n) = log(Pguess./P(1,t-1,n));
mP(2:S,t,n)  = mP(1:S-1,t-1,n) + gamma(2:S,1,n) .* (dlogP(1,t,n) - mP(1:S-1,t-1,n));
[P_ord,index] = sort(beta*phi*(exp(mD(:,t,n))*D(1,t,n)+exp(mP(:,t,n))*Pguess),'descend');
offers = lambda * D(1,t,n) /Pguess *ff(index,1,n)/sum(ff(:,1,n));
cum_offers = cumsum(offers);

if  cum_offers(S)<1                         %Not enough demand
    Pimplied = P(1,t-1,n);
else
    offer_cutoff_index = find(cum_offers>=1,1);
    Pimplied = P_ord(offer_cutoff_index);
end

Z = Pguess - Pimplied;








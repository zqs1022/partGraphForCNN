function value=getLogGauss_(mu,rho)
value=-sum(mu.^2,1)./(2*sum(rho.^2,1));

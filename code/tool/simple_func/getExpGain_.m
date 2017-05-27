function gain=getExpGain_(mu_loc,delta)
gain=exp(-sqrt(sum(mu_loc.^2,1))./(norm(delta)));
end

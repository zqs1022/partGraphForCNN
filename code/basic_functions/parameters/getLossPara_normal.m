function [weight,inValidX,minValue]=getLossPara_normal()
weight.local=1.5; %1.5; %0.5;
weight.geo=0.35; %1;
weight.app=0;
weight.reli=2.0; %4.0; %0.5; %0.3; (too large --> no deformation)
weight.gen=5;
weight.parent=10; %%%%% exp-gain
weight.super_loc=5; %25; %5;
inValidX=-3; %%%%%
minValue=-1000000;
weight.nodeNum_range=0.4;
weight.nodeNum_weight=linspace(1,1,9); %linspace(2,1,9);
weight.delta=70;

weight.parentNum=15; %%% 5;
end

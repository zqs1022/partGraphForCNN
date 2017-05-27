function [map,best_est]=produceMap(best_est,theConf,pHW_avg,mapSize,getLossPara)
[weight,~,minValue]=getLossPara();
%%% get valid votes (v,p)
v=[];
detected=[];
pHW_part=[];
nodeWeight=[];
for layer=length(best_est):-1:1
    for d=1:length(best_est(layer).depth)
        best=best_est(layer).depth(d).best;
        if(isempty(best))
            continue;
        end
        if(sum(best.GenL<=minValue)>0)
            error('errors in function produceMap');
        end
        tmp_n=length(best.GenL);
        v(end+1:end+tmp_n)=best.GenL+best.reli.*weight.reli;
        detected(end+1:end+tmp_n)=(best.local>0);
        pHW_part(:,end+1:end+tmp_n)=best.pHW_part;
        nodeWeight(:,end+1:end+tmp_n)=weight.nodeNum_weight(layer);
    end
end
nodeWeight=nodeWeight./sum(nodeWeight);
p=round(pHW_part);
%%% produce map
map=struct('map',zeros(mapSize),'maxValue',[],'MEAN_GenDisL_est',[],'position',[],'scale',[]);
valid=zeros(mapSize);




range=round(theConf.convnet.SearchRange/6);
%range=round(theConf.convnet.SearchRange/15);
%range=0;



%delta=pHW_avg.scale./2; %%%%%%%%%%%%%%%%
delta=ones(2,1).*weight.delta./2; %%%%%%%%%%%%%%%%
tmp_w=repmat(-range:range,[2*range+1,1]);
tmp_h=tmp_w';
deltaHW=[tmp_h(1:end);tmp_w(1:end)];
valueHW=getExpGain_(deltaHW,delta).*weight.super_loc;
num=size(p,2);
theMin=min(valueHW);
for i=1:num
    if(detected(i)==0)
        continue;
    end
    theP=repmat(p(:,i),[1,(2*range+1)^2])+deltaHW;
    list=find((min(theP,[],1)>0).*(theP(1,:)<=mapSize(1)).*(theP(2,:)<=mapSize(2))>0);
    theP=theP(:,list);
    position=HW2List_(theP,mapSize(1));
    theV=valueHW(list)+v(i);
    theV=theV-(theMin+v(i)); %%%%%%%%%%%%%%%%%
    map.map(position)=map.map(position)+theV'.*nodeWeight(i);
    valid(position)=1;
end
if(sum(sum(valid==0))*sum(sum(valid==1))>0)
    map.map(valid==0)=min(map.map(valid==1));
end
[map.maxValue,tmp]=max(map.map(1:end));
w=ceil(tmp/mapSize(1));
h=tmp-(w-1)*mapSize(1);
TheP=[h;w];
TheS=pHW_avg.scale;
hmin=round(max(TheP(1)-TheS(1)/2,1));
hmax=round(min(TheP(1)+TheS(1)/2,mapSize(1)));
wmin=round(max(TheP(2)-TheS(2)/2,1));
wmax=round(min(TheP(2)+TheS(2)/2,mapSize(2)));
map.position=[hmin+hmax;wmin+wmax]./2;
map.scale=[hmax-hmin+1;wmax-wmin+1];
%%% estimate DisL_est, GenDisL_est
mu_loc=repmat(map.position,[1,size(pHW_part,2)])-pHW_part;
super_loc=getLogGauss_(mu_loc,delta); %%%%%%%%%%%%
DisL_est=super_loc.*weight.super_loc;
GenDisL_est=DisL_est+v;
%map.MEAN_GenDisL_est=mean(GenDisL_est);
map.MEAN_GenDisL_est=GenDisL_est*(nodeWeight');
map.MEAN_GenReliL_est=v*(nodeWeight');
map.pHW_part=pHW_part;
map.nodeWeight=nodeWeight;
c=0;
for layer=length(best_est):-1:1
    for d=1:length(best_est(layer).depth)
        best=best_est(layer).depth(d).best;
        if(isempty(best))
            continue;
        end
        tmp_n=length(best.GenL);
        best_est(layer).depth(d).best.DisL_est=DisL_est(c+1:c+tmp_n);
        best_est(layer).depth(d).best.GenDisL_est=GenDisL_est(c+1:c+tmp_n);
        c=c+tmp_n;
    end
end
end

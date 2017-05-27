function parts=predictAOG(part_a,part_b,label_a,label_b,t_candidates,theConf,Loss)
maxNodeDist=30;

dbstop if error
%%% parts initialization
candiateNum=length(t_candidates);
layerNum=length(part_a.layer);
if(length(part_b.layer)~=layerNum)
    error('part_a and part_b does not match.');
end
Layer_tmp=repmat(struct('depth',[]),[1,layerNum]);
for layer=1:layerNum
    depNum=length(part_a.layer(layer).depth);
    if(length(part_b.layer(layer).depth)~=depNum)
        error('part_a and part_b does not match.');
    end
    Layer_tmp(layer).depth=repmat(struct('validNum',0,'nodeRank',[],'DeltaHW',[],'HoG',[],'parent','[]','LabelNum',[],'w',[]),[1,depNum]);
end
parts=repmat(struct('layer',Layer_tmp),[1,candiateNum]);
clear Layer_tmp
%%% generating parts
for layer=layerNum:-1:1
    xh=sqrt(size(Loss(layer).Gen.SUM,1));
    if(layer<layerNum)
        xh_parent=sqrt(size(Loss(layer+1).Gen.SUM,1));
        parent_depth=current_parent_depth;
    else
        xh_parent=[];
        parent_depth=[];
    end
    depNum=length(part_a.layer(layer).depth);
    current_parent_depth=repmat(struct('validNum',0,'nodeRank',[]),[1,depNum]);
    for dep=1:depNum
        validNum_a=part_a.layer(layer).depth(dep).validNum;
        validNum_b=part_b.layer(layer).depth(dep).validNum;
        if((validNum_a==0)&&(validNum_b==0))
            continue;
        end
        nodesDeltaHW_a=part_a.layer(layer).depth(dep).DeltaHW(:,1:validNum_a);
        nodesDeltaHW_b=part_b.layer(layer).depth(dep).DeltaHW(:,1:validNum_b);
        [match,TheParent,current_parent_depth(dep)]=identicalNodes(nodesDeltaHW_a,nodesDeltaHW_b,label_a,label_b,maxNodeDist,xh,xh_parent,layer,layerNum,theConf,parent_depth);
        if(sum((t_candidates>1)+(t_candidates<0))>0)
            error('Errors in variable t_candidates');
        end
        matchNum=size(match,2);
        for cand=1:candiateNum
            t=t_candidates(cand);
            tarCenter=label_a.pHW_center.*(1-t)+label_b.pHW_center.*t;
            DeltaHW_a=nodeMapping(label_a,label_b,nodesDeltaHW_a,t);
            DeltaHW_b=nodeMapping(label_b,label_a,nodesDeltaHW_b,1-t);
            [xHW_share,xHW_a,xHW_b]=getThreeTypeNodes(t,DeltaHW_a,DeltaHW_b,match,label_a,label_b,xh,layer,theConf);
            xHW_valid=[xHW_share,xHW_a,xHW_b];
            validNodeRank=HW2List_(xHW_valid,xh);
            pHW_valid=x2p_(xHW_valid,layer,theConf);
            validNum=size(xHW_valid,2);
            num_a=size(xHW_a,2);
            num_b=size(xHW_b,2);
            DeltapHW_valid=repmat(tarCenter,[1,validNum])-pHW_valid;
            w=ones(validNum,1);
            if(num_a>0)
                w(matchNum+1:matchNum+num_a)=min(1-t,0.99);
            end
            if(num_b>0)
                w(matchNum+num_a+1:matchNum+num_a+num_b)=min(t,0.99);
            end
            parts(cand).layer(layer).depth(dep).validNum=validNum;
            parts(cand).layer(layer).depth(dep).nodeRank=validNodeRank;
            parts(cand).layer(layer).depth(dep).DeltaHW=DeltapHW_valid;
            parts(cand).layer(layer).depth(dep).LabelNum=1;
            parts(cand).layer(layer).depth(dep).HoG=zeros(0,validNum);
            parts(cand).layer(layer).depth(dep).w=w;
            parts(cand).layer(layer).depth(dep).parent=TheParent;
        end
    end
end
end


function [xHW_share,xHW_a,xHW_b]=getThreeTypeNodes(t,DeltaHW_a,DeltaHW_b,match,label_a,label_b,xh,layer,theConf)
tarCenter=label_a.pHW_center.*(1-t)+label_b.pHW_center.*t;
matchNum=size(match,2);
validNum_a=size(DeltaHW_a,2);
validNum_b=size(DeltaHW_b,2);
if(matchNum>0)
    DeltaHW_share=DeltaHW_a(:,match(1,:)).*(1-t)+DeltaHW_b(:,match(2,:)).*t;
	xHW_share=p2x_(-DeltaHW_share+repmat(tarCenter,[1,matchNum]),layer,theConf,xh);
else
	xHW_share=[];
end
tmp_a=setdiff(1:validNum_a,match(1,:));
tmp_b=setdiff(1:validNum_b,match(2,:));
num_a=length(tmp_a);
num_b=length(tmp_b);
if(num_a>0)
    xHW_a=p2x_(-DeltaHW_a(:,tmp_a)+repmat(tarCenter,[1,num_a]),layer,theConf,xh);
else
	xHW_a=[];
end
if(num_b>0)
	xHW_b=p2x_(-DeltaHW_b(:,tmp_b)+repmat(tarCenter,[1,num_b]),layer,theConf,xh);
else
	xHW_b=[];
end
end


function [match,TheParent,current_depth]=identicalNodes(nodesDeltaHW_a,nodesDeltaHW_b,label_a,label_b,maxNodeDist,xh,xh_parent,layer,layerNum,theConf,parent_depth)
t=0.5;
DeltaHW_a=nodeMapping(label_a,label_b,nodesDeltaHW_a,t);
DeltaHW_b=nodeMapping(label_b,label_a,nodesDeltaHW_b,1-t);
match=uniqueNodes(DeltaHW_a,DeltaHW_b,maxNodeDist);
[xHW_share,xHW_a,xHW_b]=getThreeTypeNodes(t,DeltaHW_a,DeltaHW_b,match,label_a,label_b,xh,layer,theConf);
xHW_valid=[xHW_share,xHW_a,xHW_b];
validNodeRank=HW2List_(xHW_valid,xh);
current_depth.validNum=size(xHW_share,2);
current_depth.nodeRank=validNodeRank(1:current_depth.validNum);
if(layer==layerNum)
    tmpParent.DepID=[];
    tmpParent.DeltaHW=[];
    validNum=length(validNodeRank);
    TheParent=repmat(tmpParent,[1,validNum]);
else
    TheParent=updateParent(parent_depth,validNodeRank,layer,theConf,xh,xh_parent);
end
end


function resultsDeltaHW=nodeMapping(label_a,label_b,nodesDeltaHW,tau)
[Ha,Wa,~]=size(label_a.I);
[Hb,Wb,~]=size(label_b.I);
A1=label_a.pHW_scale./2;
A2=-label_a.pHW_scale./2;
A3=label_a.pHW_center;
A4=-[Ha;Wa]+label_a.pHW_center;
B1=label_b.pHW_scale./2;
B2=-label_b.pHW_scale./2;
B3=label_b.pHW_center;
B4=-[Hb;Wb]+label_b.pHW_center;
resultsDeltaHW=zeros(2,size(nodesDeltaHW,2));
resultsDeltaHW(1,:)=oneDimensionalZooming(nodesDeltaHW(1,:),A1(1),A2(1),A3(1),A4(1),B1(1),B2(1),B3(1),B4(1),tau);
resultsDeltaHW(2,:)=oneDimensionalZooming(nodesDeltaHW(2,:),A1(2),A2(2),A3(2),A4(2),B1(2),B2(2),B3(2),B4(2),tau);
end


function x_new=oneDimensionalZooming(x,A1,A2,A3,A4,B1,B2,B3,B4,tau) %%% x=DeltaHW(i,:)
alpha=(B2-B1)/(A2-A1);
%if((x<=A1)&&(x>=A2))
x_new=x.*alpha;
list=find(x<A2);
if(~isempty(list))
    x_new(list)=x_new(list).*exp(log(B4*A2/B2/A4)*(x(list)-A2)/(A4-A2));
end
list=find(x>A1);
if(~isempty(list))
    x_new(list)=x_new(list).*exp(log(B3*A1/B1/A3)*(x(list)-A1)/(A3-A1));
end
x_new=(1-tau).*x+tau.*x_new;
end


function match=uniqueNodes(nodesDeltaHW_a,nodesDeltaHW_b,maxNodeDist)
num_a=size(nodesDeltaHW_a,2);
num_b=size(nodesDeltaHW_b,2);
dist=zeros(num_a,num_b);
for i=1:2
    dist=dist+(repmat(nodesDeltaHW_a(i,:)',[1,num_b])-repmat(nodesDeltaHW_b(i,:),[num_a,1])).^2;
end
dist=sqrt(dist);
match=zeros(2,min(num_a,num_b));
c=0;
while(true)
    [d,i2]=min(dist,[],2);
    [d,order]=sort(d);
    i1=order;
    i2=i2(order);
    [i2,order]=unique(i2,'stable');
    i1=i1(order);
    d=d(order);
    num=sum(d<maxNodeDist);
    if(num==0)
        break;
    end
    i1=i1(1:num);
    i2=i2(1:num);
    match(:,c+1:c+num)=[i1,i2]';
    c=c+num;
    dist(i1,:)=maxNodeDist+1;
    dist(:,i2)=maxNodeDist+1;
end
match=match(:,1:c);
end

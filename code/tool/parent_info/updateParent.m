function parent=updateParent(parent_depth,validList,layer,theConf,xh,xh_parent,maxParentNum)
pHW=List2pHW(validList,layer,xh,theConf);
depthNum=length(parent_depth);
num=length(validList);
parent_num_depth=zeros(depthNum,1);
for d=1:depthNum
    parent_num_depth(d)=parent_depth(d).validNum;
end
parent_num=sum(parent_num_depth);
sqrdist=zeros(parent_num,num);
Dep=zeros(parent_num,num);
ID=zeros(parent_num,num);
DeltaH=zeros(parent_num,num);
DeltaW=zeros(parent_num,num);
c=0;
for d=1:depthNum
    parent_len=parent_num_depth(d);
    if(parent_len==0)
        continue;
    end
    parent_validList=parent_depth(d).nodeRank(1:parent_len);
    pHW_parent=List2pHW(parent_validList,layer+1,xh_parent,theConf);
    Dep(c+1:c+parent_len,:)=d;
    ID(c+1:c+parent_len,:)=repmat((1:parent_len)',[1,num]);
    dH=repmat(pHW(1,:),[parent_len,1])-repmat(pHW_parent(1,:)',[1,num]);
    dW=repmat(pHW(2,:),[parent_len,1])-repmat(pHW_parent(2,:)',[1,num]);
    DeltaH(c+1:c+parent_len,:)=dH;
    DeltaW(c+1:c+parent_len,:)=dW;
    sqrdist(c+1:c+parent_len,:)=dH.^2+dW.^2;
    c=c+parent_len;
end
topNum=min(maxParentNum,parent_num);
parent(num).DepID=[];
parent(num).DeltaHW=[];
for i=1:num
    [~,order]=sort(sqrdist(:,i),'ascend');
    order=order(1:topNum);
    parent(i).DepID=[Dep(order,i),ID(order,i)]';
    parent(i).DeltaHW=[DeltaH(order,i),DeltaW(order,i)]';
end
end


function pHW=List2pHW(validList,layer,xh,theConf)
geo=List2Geo_(validList,layer,xh,theConf);
pHW=x2p_(geo.xHW,layer,theConf);
end

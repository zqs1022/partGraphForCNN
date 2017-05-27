function parent=getParentAdvice(part,parent_record)
listLen=length(part.nodeRank);
if(listLen==0)
    parent.pHW=[];
    parent.valid=[];
    return;
end
parentNum=size(part.parent(1).DepID,2);
if((~isempty(parent_record))&&(parentNum>0))
    [pHW,valid]=getParentAdvice_mex(part.parent,parent_record,parentNum);
    %[pHW,valid]=getParentAdvice_matlab(part.parent,parent_record,parentNum);
    parent.pHW=reshape(pHW,[2,parentNum,listLen]);
    parent.valid=reshape(valid,[parentNum,listLen]);
else
    parent.pHW=zeros(2,parentNum,listLen);
    parent.valid=zeros(parentNum,listLen);
end
end


function [pHW,valid]=getParentAdvice_matlab(part_parent,parent_record,parentNum)
listLen=length(part_parent);
pHW=zeros(2,parentNum,listLen);
valid=zeros(parentNum,listLen);
for i=1:listLen
    DepID=part_parent(i).DepID;
    ThePHW=part_parent(i).DeltaHW;
    for j=1:parentNum
        Dep=DepID(1,j);
        ID=DepID(2,j);
        if(parent_record(Dep).valid(ID)==true)
            pHW(:,j,i)=parent_record(Dep).pHW(:,ID)+ThePHW(:,j);
            valid(j,i)=true;
        end
    end
end
end

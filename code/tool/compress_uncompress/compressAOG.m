function parts_c=compressAOG(parts)
parts_c=repmat(struct('layer',[]),size(parts));
partNum=length(parts);
for partID=1:partNum
    layerNum=length(parts(partID).layer);
    parts_c(partID).layer=repmat(struct('patternNum',[],'validNum',[],'LabelNum',[],'nodeRank',[],'DeltaHW',[],'parentNum',[],'p_DepID',[],'p_DeltaHW',[],'HoG',[],'HoG_min',[],'HoG_max',[]),[1,layerNum]);
    for layer=1:layerNum
        depthNum=length(parts(partID).layer(layer).depth);
        patternNum=zeros(depthNum,1,'uint16');
        LabelNum=zeros(depthNum,1,'uint16');
        validNum=zeros(depthNum,1,'uint16');
        HoGDim=0;
        for d=1:depthNum
            patternNum(d)=length(parts(partID).layer(layer).depth(d).nodeRank);
            LabelNum(d)=parts(partID).layer(layer).depth(d).LabelNum;
            validNum(d)=parts(partID).layer(layer).depth(d).validNum;
            if((HoGDim==0)&&(validNum(d)>0))
                HoGDim=size(parts(partID).layer(layer).depth(d).HoG,1);
            end
        end
        N=sum(double(patternNum));
        nodeRank=zeros(N,1,'int16');
        DeltaHW=zeros(2,N,'single');
        HoG=zeros(HoGDim,N,'single');
        parentNum=zeros(N,1,'int16');
        c=0;
        for d=1:depthNum
            t=double(patternNum(d));
            if(t>0)
                nodeRank(c+1:c+t)=int16(parts(partID).layer(layer).depth(d).nodeRank);
                DeltaHW(:,c+1:c+t)=single(parts(partID).layer(layer).depth(d).DeltaHW);
                HoG(:,c+1:c+t)=single(parts(partID).layer(layer).depth(d).HoG);
                for i=1:t
                    parentNum(c+i)=size(parts(partID).layer(layer).depth(d).parent(i).DepID,2);
                end
            end
            c=c+t;
        end
        c=0;
        cc=0;
        totalN=sum(double(parentNum));
        p_DepID=zeros(2,totalN,'int16');
        p_DeltaHW=zeros(2,totalN,'int16');
        for d=1:depthNum
            t=double(patternNum(d));
            if(t>0)
                for i=1:t
                    tt=double(parentNum(c+i));
                    p_DepID(:,cc+1:cc+tt)=int16(parts(partID).layer(layer).depth(d).parent(i).DepID);
                    p_DeltaHW(:,cc+1:cc+tt)=int16(parts(partID).layer(layer).depth(d).parent(i).DeltaHW);
                    cc=cc+tt;
                end
            end
            c=c+t;
        end
        parts(partID).layer(layer).depth=[];
        HoG_min=min(HoG,[],2);
        HoG_max=max(HoG,[],2);
        HoG=(HoG-repmat(HoG_min,[1,N]))./repmat(HoG_max-HoG_min,[1,N]);
        HoG=uint8(HoG.*255);
        parts_c(partID).layer(layer).patternNum=patternNum;
        parts_c(partID).layer(layer).validNum=validNum;
        parts_c(partID).layer(layer).LabelNum=LabelNum;
        parts_c(partID).layer(layer).nodeRank=nodeRank;
        parts_c(partID).layer(layer).DeltaHW=DeltaHW;
        parts_c(partID).layer(layer).HoG_min=HoG_min;
        parts_c(partID).layer(layer).HoG_max=HoG_max;
        parts_c(partID).layer(layer).HoG=HoG;
        parts_c(partID).layer(layer).parentNum=parentNum;
        parts_c(partID).layer(layer).p_DepID=p_DepID;
        parts_c(partID).layer(layer).p_DeltaHW=p_DeltaHW;
    end
end
end

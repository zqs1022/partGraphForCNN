function parts=uncompressAOG(parts_c)
if(isfield(parts_c(1).layer(1),'depth'))
    parts=parts_c;
    return;
end
parts=repmat(struct('layer',[]),size(parts_c));
partNum=length(parts_c);
for partID=1:partNum
    layerNum=length(parts_c(partID).layer);
    parts(partID).layer=repmat(struct('depth',[]),[1,layerNum]);
    for layer=1:layerNum
        patternNum=double(parts_c(partID).layer(layer).patternNum);
        validNum=double(parts_c(partID).layer(layer).validNum);
        LabelNum=double(parts_c(partID).layer(layer).LabelNum);
        nodeRank=double(parts_c(partID).layer(layer).nodeRank);
        DeltaHW=double(parts_c(partID).layer(layer).DeltaHW);
        try
            HoG=double(parts_c(partID).layer(layer).HoG);
            HoG_min=parts_c(partID).layer(layer).HoG_min;
            HoG_max=parts_c(partID).layer(layer).HoG_max;
            HoG=HoG.*repmat((HoG_max-HoG_min)./255,[1,size(HoG,2)])+repmat(HoG_min,[1,size(HoG,2)]);
        catch
            HoG=zeros(0,length(parts_c(partID).layer(layer).nodeRank));
        end
        parentNum=double(parts_c(partID).layer(layer).parentNum);
        p_DepID=double(parts_c(partID).layer(layer).p_DepID);
        p_DeltaHW=double(parts_c(partID).layer(layer).p_DeltaHW);
        depthNum=length(patternNum);
        parts(partID).layer(layer).depth=repmat(struct('validNum',[],'nodeRank',[],'DeltaHW',[],'HoG',[],'LabelNum',[],'parent',[]),[1,depthNum]);
        c=0;
        cc=0;
        for d=1:depthNum
            parts(partID).layer(layer).depth(d).validNum=validNum(d);
            parts(partID).layer(layer).depth(d).LabelNum=LabelNum(d);
            n=patternNum(d);
            if(n>0)
                parts(partID).layer(layer).depth(d).nodeRank=nodeRank(c+1:c+n);
                parts(partID).layer(layer).depth(d).DeltaHW=DeltaHW(:,c+1:c+n);
                parts(partID).layer(layer).depth(d).HoG=HoG(:,c+1:c+n);
                zeros(0,n);
                parent=repmat(struct('DepID',[],'DeltaHW',[]),[1,n]);
                for i=1:n
                    t=parentNum(c+i);
                    if(t>0)
                        parent(i).DepID=p_DepID(:,cc+1:cc+t);
                        parent(i).DeltaHW=p_DeltaHW(:,cc+1:cc+t);
                    end
                    cc=cc+t;
                end
                parts(partID).layer(layer).depth(d).parent=parent;
            end
            c=c+n;
        end
    end
end
end

function pHW_avg=get_pHW_avg(label_set,poseID)
pHW_avg.scale=[0;0];
pHW_avg.center=[0;0];
c=0;
for i=1:length(label_set)
    if(label_set(i).poseID==poseID)
        c=c+1;
        pHW_avg.scale=pHW_avg.scale+label_set(i).pHW_scale;
        pHW_avg.center=pHW_avg.center+label_set(i).pHW_center;
    end
end
pHW_avg.scale=pHW_avg.scale./c;
pHW_avg.center=pHW_avg.center./c;

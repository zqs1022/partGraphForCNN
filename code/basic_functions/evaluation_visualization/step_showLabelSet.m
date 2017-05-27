function step_showLabelSet(label_set)
list=[];
for i=1:length(label_set)
    if(~isempty(label_set(i).obj))
        list=[list,i];
    end
end
label_set=label_set(list);
num=length(label_set);
PoseID=zeros(num,1);
for i=1:num
    PoseID(i)=label_set(i).poseID;
end
maxPoseID=max(PoseID);
for i=0:maxPoseID
    list=find(PoseID'==i);
    fprintf('Pose %d\n',i)
    for tar=list
        figure(tar);
        imshow(uint8(label_set(tar).I));
        hold on
        label=label_set(tar);
        tar_h=label.pHW_center(1);
        tar_w=label.pHW_center(2);
        scale_h=label.pHW_scale(1);
        scale_w=label.pHW_scale(2);
        plot(tar_w+[0.5,0.5,-0.5,-0.5,0.5].*scale_w,tar_h+[0.5,-0.5,-0.5,0.5,0.5].*scale_h);
        pause;
        pause(0.5);
        close gcf
        pause(0.5);
    end
end
end

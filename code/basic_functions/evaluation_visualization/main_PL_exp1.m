function main_PL_exp1
poseNum=1;
sampleNum_perPose=2; %[2,3,4,5];
partNum=1;

results=zeros(30,partNum,length(sampleNum_perPose));
% results(1,:,:)=do_baseline_part('n01443537',partNum,poseNum,sampleNum_perPose);
% results(2,:,:)=do_baseline_part('n01503061',partNum,poseNum,sampleNum_perPose);
% results(3,:,:)=do_baseline_part('n01639765',partNum,poseNum,sampleNum_perPose);
% results(4,:,:)=do_baseline_part('n01662784',partNum,poseNum,sampleNum_perPose);
% results(5,:,:)=do_baseline_part('n01674464',partNum,poseNum,sampleNum_perPose);
% results(6,:,:)=do_baseline_part('n01882714',partNum,poseNum,sampleNum_perPose);
% results(7,:,:)=do_baseline_part('n01982650',partNum,poseNum,sampleNum_perPose);
% results(8,:,:)=do_baseline_part('n02084071',partNum,poseNum,sampleNum_perPose);
results(9,:,:)=do_baseline_part('n02118333',partNum,poseNum,sampleNum_perPose);
% results(10,:,:)=do_baseline_part('n02121808',partNum,poseNum,sampleNum_perPose);
% results(11,:,:)=do_baseline_part('n02129165',partNum,poseNum,sampleNum_perPose);
% results(12,:,:)=do_baseline_part('n02129604',partNum,poseNum,sampleNum_perPose);
% results(13,:,:)=do_baseline_part('n02131653',partNum,poseNum,sampleNum_perPose);
% results(14,:,:)=do_baseline_part('n02324045',partNum,poseNum,sampleNum_perPose);
% results(15,:,:)=do_baseline_part('n02342885',partNum,poseNum,sampleNum_perPose);
% results(16,:,:)=do_baseline_part('n02355227',partNum,poseNum,sampleNum_perPose);
% results(17,:,:)=do_baseline_part('n02374451',partNum,poseNum,sampleNum_perPose);
% results(18,:,:)=do_baseline_part('n02391049',partNum,poseNum,sampleNum_perPose);
% results(19,:,:)=do_baseline_part('n02395003',partNum,poseNum,sampleNum_perPose);
% results(20,:,:)=do_baseline_part('n02398521',partNum,poseNum,sampleNum_perPose);
% results(21,:,:)=do_baseline_part('n02402425',partNum,poseNum,sampleNum_perPose);
% results(22,:,:)=do_baseline_part('n02411705',partNum,poseNum,sampleNum_perPose);
% results(23,:,:)=do_baseline_part('n02419796',partNum,poseNum,sampleNum_perPose);
% results(24,:,:)=do_baseline_part('n02437136',partNum,poseNum,sampleNum_perPose);
% results(25,:,:)=do_baseline_part('n02444819',partNum,poseNum,sampleNum_perPose);
% results(26,:,:)=do_baseline_part('n02454379',partNum,poseNum,sampleNum_perPose);
% results(27,:,:)=do_baseline_part('n02484322',partNum,poseNum,sampleNum_perPose);
% results(28,:,:)=do_baseline_part('n02503517',partNum,poseNum,sampleNum_perPose);
% results(29,:,:)=do_baseline_part('n02509815',partNum,poseNum,sampleNum_perPose);
% results(30,:,:)=do_baseline_part('n02510455',partNum,poseNum,sampleNum_perPose);
results=results.*100;
for i=1:partNum
    disp(reshape(results(:,i,:),[size(results,1),length(sampleNum_perPose)]));
end


% partID=1;
% annotateList=[1,2,9,10,18,26,28,46,52,66;15,24,27,51,56,69,72,78,197,231;114,137,148,152,190,240,310,333,402,550];
% flipList=[0,0,0,0,1,1,1,0,0,0;0,1,0,1,0,0,1,0,0,0;1,0,1,1,0,0,1,1,1,0];
% step_batch_pre('n01443537',partID,annotateList,flipList);

end


function results=do_baseline_part(Name_batch,partNum,poseNum,sampleNum_perPose)
global conf;
% close all;
% load([conf.output.dir,'Exp1/',Name_batch,'/all_statistics.mat'],'stat_all');
% load([conf.output.dir,'Exp1/',Name_batch,'/net_finetune.mat'],'net','info');
% folderName=sprintf('%sExp2/%s/',conf.output.dir,Name_batch);
% mkdir(folderName);
% for partID=1:partNum
%     sNum=max(sampleNum_perPose);
%     mkdir([conf.output.dir,Name_batch]);
%     fileName_ant=sprintf('%s%s/part%02d_ant.mat',conf.output.dir,Name_batch,partID);
%     if(exist(fileName_ant,'file')==0)
%         label_set=getLabelSet(Name_batch,partID,poseNum,sNum,conf,net,stat_all);
%         save(fileName_ant,'label_set');
%     end
% end
% mkdir([conf.output.dir,Name_batch]);
% fileN='all_statistics.mat';copyfile([conf.output.dir,'Exp1/',Name_batch,'/',fileN],[conf.output.dir,Name_batch,'/',fileN]);
% fileN='genLoss.mat';copyfile([conf.output.dir,'Exp1/',Name_batch,'/',fileN],[conf.output.dir,Name_batch,'/',fileN]);
% fileN='object_dist.mat';copyfile([conf.output.dir,'Exp1/',Name_batch,'/',fileN],[conf.output.dir,Name_batch,'/',fileN]);
% fileN='net*.*';copyfile([conf.output.dir,'Exp1/',Name_batch,'/',fileN],[conf.output.dir,Name_batch,'/']);
% fileN='truth_*.*';copyfile([conf.output.dir,'Exp1/',Name_batch,'/',fileN],[conf.output.dir,Name_batch,'/']);
% for partID=1:partNum
%     fileName_ant=sprintf('%s%s/part%02d_ant.mat',conf.output.dir,Name_batch,partID);
%     clear label_set;
%     load(fileName_ant,'label_set');
%     label_set_large=label_set;clear label_set;
%     for i=1:length(sampleNum_perPose)
%         sNum=sampleNum_perPose(i);
%         label_set=label_set_large(1:sNum*poseNum);
%         step_batch(Name_batch,partID,label_set,poseNum);
%         fileName_s=sprintf('%s%s/part%02d_step*.mat',conf.output.dir,Name_batch,partID);
%         folderName_d=sprintf('%sExp2/%s/sampleNum_%02d/',conf.output.dir,Name_batch,sNum);
%         mkdir(folderName_d);
%         movefile(fileName_s,folderName_d);
%     end
% end
results=zeros(partNum,length(sampleNum_perPose));
for partID=1:partNum
    for i=1:length(sampleNum_perPose)
        results(partID,i)=step_getResult(Name_batch,partID,Name_batch);
    end
end
end


function label_set=getLabelSet(Name_batch,partID,poseNum,sNum,theConf,theNet,theStat_all)
tarSize=[224,224];
fileName=sprintf('%sExp1/%s/part%02d_step%02d.mat',theConf.output.dir,Name_batch,partID,poseNum);
load(fileName,'label_set','parts','inference');
parts=uncompressAOG(parts);
if((length(label_set)~=poseNum)||(length(parts)~=poseNum)||(sNum<=1))
    error('Errors in function getLabelSet.');
end
clear parts
label_set(sNum*poseNum).app=[];
objNum=length(inference);
for step=1:poseNum
    objValue=zeros(objNum,1);
    for i=1:objNum
        objValue(i)=inference(i).poseGenDisL(step);
    end
    [~,order]=sort(objValue,'descend');
    objset=readAnnotation(Name_batch,theConf);
    if(length(objset)~=objNum)
        error('Errors in function getLabelSet.');
    end
    c=1;
    for i=2:length(order)
        obj=objset(order(i));
        figure;
        subplot(1,2,1);
        [I_patch,~]=getI(label_set(step).obj,theConf,tarSize,false);
        imshow(uint8(I_patch));
        subplot(1,2,2)
        [I_patch,~]=getI(obj,theConf,tarSize,false);
        imshow(uint8(I_patch));
        IsFlip=getYNInput('Is it a flipped pose ? [y/n] or other strings for occlusions ');
        close gcf;
        if(IsFlip==-1)
            continue;
        end
        c=c+1;
        idx=(c-1)*poseNum+step;
        label=label_part(obj,theConf,theNet,theStat_all,IsFlip);
        label.poseID=step;
        label.flip=IsFlip;
        label_set(idx)=label;
        if(c==sNum)
            break;
        end
    end
end
end


function a=getYNInput(str)
str=[str,'   '];
while(true)
    yn=input(str,'s');
    if(strcmp(yn,'y'))
        a=true;
        break;
    else
        if(strcmp(yn,'n'))
            a=false;
            break;
        else
            a=-1;
            break;
        end
    end
end
end

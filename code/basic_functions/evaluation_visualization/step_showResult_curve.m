function step_showResult_curve
load('./mat/n01443537/genLoss.mat','Loss');
layerNum=length(Loss);
global xh;
xh=zeros(layerNum,1);
for layer=1:layerNum
    xh(layer)=sqrt(size(Loss(layer).Gen.SUM,1));
end


% partID=1;
% trainSampleNum=30;
% root_tar='../../../../home/vcla/zqs/QAinDeep_vclagpu/code/mat/'; %%%%%%%%%%%
% paraFolder='Para_3.5';
% normDist=zeros(30,trainSampleNum);
% IoURate=zeros(30,trainSampleNum);
% LOCRate=zeros(30,trainSampleNum);
% [normDist(1,:),IoURate(1,:),LOCRate(1,:)]=do_baseline_part('n01443537',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(2,:),IoURate(2,:),LOCRate(2,:)]=do_baseline_part('n01503061',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(3,:),IoURate(3,:),LOCRate(3,:)]=do_baseline_part('n01639765',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(4,:),IoURate(4,:),LOCRate(4,:)]=do_baseline_part('n01662784',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(5,:),IoURate(5,:),LOCRate(5,:)]=do_baseline_part('n01674464',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(6,:),IoURate(6,:),LOCRate(6,:)]=do_baseline_part('n01882714',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(7,:),IoURate(7,:),LOCRate(7,:)]=do_baseline_part('n01982650',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(8,:),IoURate(8,:),LOCRate(8,:)]=do_baseline_part('n02084071',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(9,:),IoURate(9,:),LOCRate(9,:)]=do_baseline_part('n02118333',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(10,:),IoURate(10,:),LOCRate(10,:)]=do_baseline_part('n02121808',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(11,:),IoURate(11,:),LOCRate(11,:)]=do_baseline_part('n02129165',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(12,:),IoURate(12,:),LOCRate(12,:)]=do_baseline_part('n02129604',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(13,:),IoURate(13,:),LOCRate(13,:)]=do_baseline_part('n02131653',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(14,:),IoURate(14,:),LOCRate(14,:)]=do_baseline_part('n02324045',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(15,:),IoURate(15,:),LOCRate(15,:)]=do_baseline_part('n02342885',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(16,:),IoURate(16,:),LOCRate(16,:)]=do_baseline_part('n02355227',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(17,:),IoURate(17,:),LOCRate(17,:)]=do_baseline_part('n02374451',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(18,:),IoURate(18,:),LOCRate(18,:)]=do_baseline_part('n02391049',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(19,:),IoURate(19,:),LOCRate(19,:)]=do_baseline_part('n02395003',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(20,:),IoURate(20,:),LOCRate(20,:)]=do_baseline_part('n02398521',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(21,:),IoURate(21,:),LOCRate(21,:)]=do_baseline_part('n02402425',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(22,:),IoURate(22,:),LOCRate(22,:)]=do_baseline_part('n02411705',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(23,:),IoURate(23,:),LOCRate(23,:)]=do_baseline_part('n02419796',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(24,:),IoURate(24,:),LOCRate(24,:)]=do_baseline_part('n02437136',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(25,:),IoURate(25,:),LOCRate(25,:)]=do_baseline_part('n02444819',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(26,:),IoURate(26,:),LOCRate(26,:)]=do_baseline_part('n02454379',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(27,:),IoURate(27,:),LOCRate(27,:)]=do_baseline_part('n02484322',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(28,:),IoURate(28,:),LOCRate(28,:)]=do_baseline_part('n02503517',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(29,:),IoURate(29,:),LOCRate(29,:)]=do_baseline_part('n02509815',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(30,:),IoURate(30,:),LOCRate(30,:)]=do_baseline_part('n02510455',partID,root_tar,trainSampleNum,paraFolder);
% disp(normDist);
% disp(IoURate);
% disp(LOCRate);
% disp(mean(normDist,1));


paraFolder='Para_3.5'; %'Para_7.0'; %'Para_1.7'; 'Para_3.5';
partID=1; %1;
trainSampleNum=30;
root_tar='../../../../home/vcla/zqs/QAinDeep_vclagpu/code/mat/'; %%%%%%%%%%%
normDist=zeros(6,trainSampleNum);
IoURate=zeros(6,trainSampleNum);
LOCRate=zeros(6,trainSampleNum);
childNum=zeros(6,trainSampleNum,3);
[normDist(1,:),IoURate(1,:),LOCRate(1,:),childNum(1,:,:)]=do_baseline_part('bird',partID,root_tar,trainSampleNum,paraFolder);
[normDist(2,:),IoURate(2,:),LOCRate(2,:),childNum(2,:,:)]=do_baseline_part('cat',partID,root_tar,trainSampleNum,paraFolder);
[normDist(3,:),IoURate(3,:),LOCRate(3,:),childNum(3,:,:)]=do_baseline_part('cow',partID,root_tar,trainSampleNum,paraFolder);
[normDist(4,:),IoURate(4,:),LOCRate(4,:),childNum(4,:,:)]=do_baseline_part('dog',partID,root_tar,trainSampleNum,paraFolder);
[normDist(5,:),IoURate(5,:),LOCRate(5,:),childNum(5,:,:)]=do_baseline_part('horse',partID,root_tar,trainSampleNum,paraFolder);
[normDist(6,:),IoURate(6,:),LOCRate(6,:),childNum(6,:,:)]=do_baseline_part('sheep',partID,root_tar,trainSampleNum,paraFolder);
disp(normDist);
disp(IoURate);
disp(LOCRate);
disp(reshape(mean(childNum,1),[trainSampleNum,3]));


% partID=6;
% trainSampleNum=20;
% root_tar='./mat/';
% paraFolder='Para_3.5';
% [normDist,IoURate,LOCRate]=do_baseline_part('cub200',partID,root_tar,trainSampleNum,paraFolder);
% disp(normDist);
% disp(IoURate);
% disp(LOCRate);



% paraFolder='Para_oneByOne';
% partID=1;
% trainSampleNum=30;
% root_tar='./mat/'; %%%%%%%%%%%
% normDist=zeros(6,trainSampleNum);
% IoURate=zeros(6,trainSampleNum);
% LOCRate=zeros(6,trainSampleNum);
% childNum=zeros(6,trainSampleNum,3);
% [normDist(1,:),IoURate(1,:),LOCRate(1,:),childNum(1,:,:)]=do_baseline_part('bird',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(2,:),IoURate(2,:),LOCRate(2,:),childNum(2,:,:)]=do_baseline_part('cat',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(3,:),IoURate(3,:),LOCRate(3,:),childNum(3,:,:)]=do_baseline_part('cow',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(4,:),IoURate(4,:),LOCRate(4,:),childNum(4,:,:)]=do_baseline_part('dog',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(5,:),IoURate(5,:),LOCRate(5,:),childNum(5,:,:)]=do_baseline_part('horse',partID,root_tar,trainSampleNum,paraFolder);
% [normDist(6,:),IoURate(6,:),LOCRate(6,:),childNum(6,:,:)]=do_baseline_part('sheep',partID,root_tar,trainSampleNum,paraFolder);
% disp(normDist);
% disp(IoURate);
% disp(LOCRate);
% disp(reshape(mean(childNum,1),[trainSampleNum,3]));


% partID=6;
% trainSampleNum=20;
% root_tar='./mat/';
% paraFolder='Para_oneByOne';
% [normDist,IoURate,LOCRate]=do_baseline_part('cub200',partID,root_tar,trainSampleNum,paraFolder);
% disp(normDist);
% disp(IoURate);
% disp(LOCRate);
end


function [normDist,IoURate,LOCRate,childNum]=do_baseline_part(Name_batch,partID,root_tar,trainSampleNum,paraFolder)
global conf;
global xh;
if(nargout>=4)
    IsChildNum=true;
else
    IsChildNum=false;
end

IsChildNum=false;

normDist=zeros(1,trainSampleNum);
IoURate=zeros(1,trainSampleNum);
LOCRate=zeros(1,trainSampleNum);
childNum=zeros(1,trainSampleNum,3);
a=load(sprintf('%s%s/%s/part%02d_curve.mat',root_tar,Name_batch,paraFolder,partID),'curve');
stepNum=length(a.curve.normDist);
pre.antNum=0;
pre.normDist=0.5;
pre.LOCRate=0;
pre.IoURate=0;
for step=1:stepNum
    filename=sprintf('%s%s/%s/part%02d_step%02d.mat',root_tar,Name_batch,paraFolder,partID,step);
    b=load(filename,'label_set');
    antNum=length(b.label_set);
    list=pre.antNum+1:min(antNum,trainSampleNum);
    cur.antNum=antNum;
    cur.normDist=a.curve.normDist(step);
    cur.LOCRate=a.curve.LOCRate(step);
    cur.IoURate=a.curve.IoURate(step);
    normDist(list)=getInterpolation(pre.normDist,cur.normDist,pre.antNum,cur.antNum,length(list));
    IoURate(list)=getInterpolation(pre.IoURate,cur.IoURate,pre.antNum,cur.antNum,length(list));
    LOCRate(list)=getInterpolation(pre.LOCRate,cur.LOCRate,pre.antNum,cur.antNum,length(list));
    pre=cur;
    if(pre.antNum>=trainSampleNum)
        break;
    end
end
if(pre.antNum<trainSampleNum)
    error('Errors here.');
end
if(IsChildNum)
    pre.antNum=0;
    pre.appNum=0;
    pre.latNum=0;
    pre.neuNum=0;
    for step=[2:3:stepNum-1,stepNum]
        filename=sprintf('%s%s/%s/part%02d_step%02d.mat',root_tar,Name_batch,paraFolder,partID,step);
        b=load(filename,'label_set','parts');
        antNum=length(b.label_set);
        list=pre.antNum+1:min(antNum,trainSampleNum);
        cur.antNum=antNum;
        [cur.appNum,cur.latNum,cur.neuNum]=getNodeNum(b.parts,xh,conf);
        tmp1=getInterpolation(pre.appNum,cur.appNum,pre.antNum,cur.antNum,length(list));
        tmp2=getInterpolation(pre.latNum,cur.latNum,pre.antNum,cur.antNum,length(list));
        tmp3=getInterpolation(pre.neuNum,cur.neuNum,pre.antNum,cur.antNum,length(list));
        childNum(1,list,1)=tmp1;
        childNum(1,list,2)=tmp2./tmp1;
        childNum(1,list,3)=tmp3./tmp2;
        pre=cur;
        if(pre.antNum>=trainSampleNum)
            break;
        end
    end
    if(pre.antNum<trainSampleNum)
        error('Errors here.');
    end
end
disp(Name_batch);
end


function value=getInterpolation(pre_v,cur_v,pre_antNum,cur_antNum,len)
value=(1:len).*((cur_v-pre_v)/(cur_antNum-pre_antNum))+pre_v;
end

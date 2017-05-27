function parts=try_interpolation
global conf;


% %%% copying images to a self-prepared folder
% Name_batch='n02084071';
% Name_batch_selfdata='dog';
% objset=readAnnotation(Name_batch,conf);
% tarSize=[224,224];
% IsFlip=false;
% for i=1:length(objset)
%     [I_patch,~]=getI(objset(i),conf,tarSize,IsFlip);
%     filename=sprintf('../data/self-prepared/%s/%05d.JPEG',Name_batch_selfdata,i);
%     imwrite(uint8(I_patch),filename);
% end
% return


Name_new='dog';
poseNum=2;
f_start=1;
f_end=10;

% load([conf.output.dir,Name_new,'/all_statistics.mat'],'stat_all');
% theStat_all=stat_all;clear stat_all;
% load([conf.output.dir,Name_new,'/net_finetune.mat'],'net','info');
% objset=readAnnotation(Name_new,conf);
% IsFlip=false;
% label_a=label_part(objset(f_start),conf,net,theStat_all,IsFlip);
% label_b=label_part(objset(f_end),conf,net,theStat_all,IsFlip);
% label_a.poseID=1;
% label_b.poseID=2;
% label_a.flip=false;
% label_b.flip=false;
% label_set=[label_a,label_b];
% step_batch('dog',1,label_set,poseNum);


t_candidates=0:0.1:1;
filename=sprintf('%s%s/part%02d_step%02d.mat',conf.output.dir,Name_new,1,2);
load(filename,'label_set','parts');
filename=sprintf('%s%s/genLoss.mat',conf.output.dir,Name_new);
load(filename,'Loss');
parts=uncompressAOG(parts);
part_a=parts(1);
part_b=parts(2);
label_a=label_set(1);
label_b=label_set(2);
clear parts
tic
parts=predictAOG(part_a,part_b,label_a,label_b,t_candidates,conf,Loss);
toc
end

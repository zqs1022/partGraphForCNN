function step_showNodes(Name_batch,modelDir,modelDir2,partID,showPercent)
if(~isempty(modelDir2))
    modelDir2=[modelDir2,'/'];
end
if(nargin<5)
    showPercent=1;
end
global conf;
theConf=conf;
stepNum=0;
while(true)
    cateLabel_name=sprintf('%s%s/%s/%spart%02d_step%02d.mat',theConf.output.dir,modelDir,Name_batch,modelDir2,partID,stepNum+1);
    if(exist(cateLabel_name,'file')==2)
        stepNum=stepNum+1;
    else
        break;
    end
end
cateLabel_name=sprintf('%s%s/%s/%spart%02d_step%02d.mat',theConf.output.dir,modelDir,Name_batch,modelDir2,partID,stepNum);
load(cateLabel_name,'parts','parts_corner','label_set');



parts=uncompressAOG(parts);
thePara=@getLossPara_normal;
% parts=uncompressAOG(parts_corner(3,:));
% thePara=@getLossPara_highParent;



layerNum=length(parts(1).layer);
partNum=length(parts);
report=zeros(partNum,layerNum);
for p=1:partNum
    for layer=1:layerNum
        theDepth=parts(p).layer(layer).depth;
        c=0;
        for d=1:length(theDepth)
            c=c+theDepth(d).validNum;
        end
        report(p,layer)=c;
    end
end
disp('Node number:');
disp(report);
file_genLoss=[theConf.output.dir,Name_batch,'/genLoss.mat'];
load(file_genLoss,'Loss');
load([theConf.output.dir,Name_batch,'/net_finetune.mat'],'net','info');
load([theConf.output.dir,Name_batch,'/all_statistics.mat'],'stat_all');
theStat_all=stat_all;clear stat_all;
c=0;



% theConf.data.imgdir=theConf.data.framedir;
% objset=readAnnotation(Name_batch,theConf); %%%%%%%%%%%%%%%%%%%
% label_set(1).obj=objset(5); %%%%%%%%%%%%%%%%%%%
% label_set(1).flip=false; %%%%%%%%%%%%%%%%%%%
% label_set(2).obj=objset(10); %%%%%%%%%%%%%%%%%%%
% label_set(2).flip=false; %%%%%%%%%%%%%%%%%%%
% label_set(3).obj=objset(15); %%%%%%%%%%%%%%%%%%%
% label_set(3).flip=false; %%%%%%%%%%%%%%%%%%%



for label=label_set
    c=c+1;
    results=step_inference_part_obj(label,label_set,parts,theConf,net,Loss,theStat_all,thePara);
    list=find(results(1).detectedList>0);
    xHWList=results(1).xHWList(:,list);
    layerList=results(1).layerList(list);
    I_patch=results(1).I_patch;
    map=results(1).map;
    poseID=results(1).poseID;
    showFilter(xHWList,layerList,I_patch,showPercent);hold on;plot(map.position(2),map.position(1),'wo');pause(0.2);
    figure;imagesc(map.map);pause(0.2);
    fprintf('step %d  object ID %d  pose ID %d\n',c,label.obj.ID,poseID);
    pause;
    close all;
end
end

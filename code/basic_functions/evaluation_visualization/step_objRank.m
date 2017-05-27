function step_objRank(Name_batch,modelDir,modelDir2,partID)
if(~isempty(modelDir2))
    modelDir2=[modelDir2,'/'];
end
global conf;
theConf=conf;
try
    load(sprintf('%s%s/truth_part%02d_small.mat',theConf.output.dir,Name_batch,partID),'truth');
    %load(sprintf('%s%s/truth_part%02d.mat',theConf.output.dir,Name_batch,partID),'truth');
catch
    truth=[];
end
iterNum=0;
while(true)
    cateLabel_name=sprintf('%s%s/%s/%spart%02d_step%02d.mat',theConf.output.dir,modelDir,Name_batch,modelDir2,partID,iterNum+1);
    if(exist(cateLabel_name,'file')==2)
        iterNum=iterNum+1;
    else
        break;
    end
end


%iterNum=28


cateLabel_name=sprintf('%s%s/%s/%spart%02d_step%02d.mat',theConf.output.dir,modelDir,Name_batch,modelDir2,partID,iterNum);
load(cateLabel_name,'label_set','inference');
objset=readAnnotation(Name_batch,theConf);
va=[];
for par=1:length(inference)
    if(isempty(inference(par).map))
        va=[va,-100000];
    else
        va=[va,inference(par).map.MEAN_GenDisL_est];
    end
end
[va,order]=sort(va,'descend');



%order=1:length(order); %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


figure;hist(va);pause
distRecord=[];
for par=1:1:length(inference)
    tar=order(par);
    if(isempty(truth)||isempty(truth(tar).pHW_center))
        continue;
    end
    obj=objset(tar);
    [I_patch,~]=getI(obj,theConf,[224,224],inference(tar).flip);
    pI=uint8(I_patch);
    tar_h=inference(tar).map.position(1);
    tar_w=inference(tar).map.position(2);
    if(isempty(truth)||isempty(truth(tar).pHW_center))
        distRate=-1;
    else
        if(inference(tar).flip)
            theP=truth(tar).pHW_center;
            theP(2,:)=size(pI,2)+1-theP(2,:);
        else
            theP=truth(tar).pHW_center;
        end
        distRate=min(sqrt(sum((theP-repmat(inference(tar).map.position,[1,size(truth(tar).pHW_center,2)])).^2,1))./(224*sqrt(2)));
        distRecord=[distRecord,distRate];
    end
    if(distRate<0.08)
        %continue;
    end
    scale=inference(tar).map.scale;
    
    
    figure;
    %set(gcf,'Visible','off');
    imshow(pI);hold on;pause(0.5);
    %for i=1:size(theP,2),plot(theP(2,i),theP(1,i),'wo','MarkerSize',10);end
    plot(tar_w+[-0.5,-0.5,0.5,0.5,-0.5].*scale(2),tar_h+[-0.5,0.5,0.5,-0.5,-0.5].*scale(1),'-','Color',[0.99,0.99,0.99],'LineWidth',5);pause(0.5);
    
    
    %filename=sprintf('./fig_out/objRank/%s_part%02d_%05d',Name_batch,partID,par);saveas(gcf,filename,'jpg');close gcf;pause(0.5);I=imread([filename,'.jpg']);I=I(81:730,271:920,:);imwrite(I,[filename,'.jpg']);continue;
    
    
    %figure;imagesc(inference(tar).map.map);pause(0.5);
    [~,poseID]=max(inference(tar).poseGenDisL);
    fprintf('#%d:  objectID %d  IsFlip %d  GenDisL %.3f  maxVote %.1f  poseID %d distRate %.2f\n',par,inference(tar).objID,inference(tar).flip,va(par),inference(tar).map.MEAN_GenDisL_est,poseID,distRate)
    pause(1.0);
    pause;
    close all;
    pause(1.5);
end
figure;hist(distRecord);
end

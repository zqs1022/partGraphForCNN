function updateInference(Name_batch,partID,net,stepList,theConf,minSize,svmWeight,getLossPara)
topPoseNum=3;

IsOnlyForTruth=true;
root_out=[theConf.output.dir,Name_batch,'/boxRegression/'];
global testingTruthFile;
testingTruthFile=sprintf('truth_part%02d_small.mat',partID);
emptyInfer=struct('map',[],'objID',[],'flip',[],'poseGenDisL',[],'posePos',[],'classResponse',[]);
objset=readAnnotation(Name_batch,theConf);
tarSize=net.meta.normalization.imageSize(1:2);
for i=length(stepList):-1:1
    step=stepList(i);
    m=load(sprintf('%spart%02d_step%02d_svm.mat',root_out,partID,step),'linearSVM');
    r=load(sprintf('%s%s/part%02d_step%02d.mat',theConf.output.dir,Name_batch,partID,step),'inference','label_set');
    objNum=length(r.inference);
    if(objNum~=length(objset))
        error('Errors here.');
    end
    validObj=validObjList(theConf,Name_batch,objNum,IsOnlyForTruth);
    for objID=1:objNum
        if(validObj(objID))
            infer=r.inference(objID);
            [I_patch,~]=getI(objset(objID),theConf,tarSize,infer.flip);
            poseNum=length(infer.poseGenDisL);
            [~,order]=sort(infer.poseGenDisL,'descend');
            order=order(1:min(topPoseNum,poseNum));
            list=setdiff(1:poseNum,order);
            infer.poseGenDisL(list)=-100000000;
            for poseID=order
                pHW_avg=get_pHW_avg(r.label_set,poseID);
                posePos=updatePoseInfer(r.inference(objID).posePos(poseID),net,uint8(I_patch),minSize,svmWeight,pHW_avg.scale,getLossPara,m,theConf);
                infer.poseGenDisL(poseID)=posePos.MEAN_GenDisL_est;
                infer.posePos(poseID)=posePos;
            end
            [~,idx]=max(infer.poseGenDisL);
            
            
            
%             figure(objID);
%             subplot(1,2,1);imshow(uint8(I_patch));hold on;
%             theH=infer.map.position(1);theW=infer.map.position(2);
%             SH=infer.map.scale(1);SW=infer.map.scale(2);
%             plot(theW+[-0.5,-0.5,0.5,0.5,-0.5].*SW,theH+[-0.5,0.5,0.5,-0.5,-0.5].*SH,'w-');
%             subplot(1,2,2);imshow(uint8(I_patch));hold on;
%             theH=infer.posePos(idx).position(1);theW=infer.posePos(idx).position(2);
%             SH=infer.posePos(idx).scale(1);SW=infer.posePos(idx).scale(2);
%             plot(theW+[-0.5,-0.5,0.5,0.5,-0.5].*SW,theH+[-0.5,0.5,0.5,-0.5,-0.5].*SH,'w-');
%             idx1=order(1);idx2=idx;
%             disp([idx1,infer.poseGenDisL(idx1),infer.posePos(idx1).MEAN_GenReliL_est;idx2,infer.poseGenDisL(idx2),infer.posePos(idx2).MEAN_GenReliL_est]);
%             pause;
%             close gcf;
%             pause(2.0);
            
            
            infer.map.MEAN_GenDisL_est=infer.poseGenDisL(idx);
            infer.map.position=infer.posePos(idx).position;
            infer.map.scale=infer.posePos(idx).scale;
            r.inference(objID)=infer;
        else
            r.inference(objID)=emptyInfer;
        end
        disp([i,objID]);
    end
    inference=r.inference;
    label_set=r.label_set;
    save(sprintf('%s/part%02d_step%02d.mat',root_out,partID,step),'inference','label_set');
    Name_batch_target=[Name_batch,'/boxRegression'];
    [curve.IoURate,curve.normDist,curve.LOCRate,~]=step_getResult(Name_batch,partID,Name_batch_target,testingTruthFile,step);
    save(sprintf('%s/part%02d_step%02d.mat',root_out,partID,step),'inference','label_set','curve');
    fprintf('normDist= %f\n',curve.normDist)
end
end


function posePos=updatePoseInfer(posePos,net,I_patch,minSize,svmWeight,scale_avg,getLossPara,model,theConf)
scaleRate=1.2;
driftList=-0.5:0.25:0.5;

[weight,~,~]=getLossPara();
latNum=length(posePos.nodeWeight);
delta=ones(2,1).*weight.delta./2; %%%%%%%%%%%%%%%%
[h,w,~]=size(I_patch);
tar_h=posePos.position(1);
tar_w=posePos.position(2);
scale_h=max(scale_avg(1),minSize);
scale_w=max(scale_avg(2),minSize);
poseGenDisL=-1000000;
range=round(theConf.convnet.SearchRange/6);
svmS=[];
for dx=driftList
    for dy=driftList
        theH=tar_h+dy*scale_h;
        theW=tar_w+dx*scale_w;
        SH=scale_h*scaleRate;
        SW=scale_w*scaleRate;
        fea=extractFC7(I_patch,net,theH,theW,SH,SW)';
        [~,tmp]=predict(model.linearSVM,fea);
        %svmScore=max(tmp(:,end),0)*svmWeight;
        svmScore=max(tmp(:,end),-0.5)*svmWeight;
        mu_loc=repmat([theH;theW],[1,latNum])-posePos.pHW_part;
        mu_loc=min(max(mu_loc,-range),range);
        %super_loc=getLogGauss_(mu_loc,delta); %%%%%%%%%%%%
        super_loc=getExpGain_(mu_loc,delta); %%%%%%%%%%%%
        DisL_est=super_loc.*weight.super_loc;
        score=svmScore+posePos.MEAN_GenReliL_est+DisL_est*(posePos.nodeWeight');
        if(poseGenDisL<score)
            poseGenDisL=score;
            minH=round(max(theH-scale_avg(1)/2,1));
            minW=round(max(theW-scale_avg(2)/2,1));
            maxH=round(min(theH+scale_avg(1)/2,h));
            maxW=round(min(theW+scale_avg(2)/2,w));
            posePos.position=round([minH+maxH;minW+maxW]./2);
            posePos.scale=[maxH-minH;maxW-minW];
            svmS=svmScore;
        end
    end
end
posePos.MEAN_GenDisL_est=poseGenDisL;
disp([poseGenDisL,svmS]);
end

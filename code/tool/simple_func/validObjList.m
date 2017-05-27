function validObj=validObjList(theConf,Name_batch,objNum,IsOnlyForTruth)
if(IsOnlyForTruth)
    global testingTruthFile;
    validObj=zeros(1,objNum);
    fileName=sprintf('%s%s/%s',theConf.output.dir,Name_batch,testingTruthFile);
    a=load(fileName,'truth');
    for i=1:objNum
        validObj(i)=~isempty(a.truth(i).obj);
    end
else
    validObj=ones(1,objNum);
end
end

function main_exp1
root_tar='Exp1/';
partNum=2;

results=zeros(30,partNum);
normDist=zeros(30,partNum);
loc=zeros(30,partNum);

[normDist(1,:,:),results(1,:,:),loc(1,:,:)]=do_baseline_part('n01443537',partNum,root_tar);
[normDist(2,:,:),results(2,:,:),loc(2,:,:)]=do_baseline_part('n01503061',partNum,root_tar);
[normDist(3,:,:),results(3,:,:),loc(3,:,:)]=do_baseline_part('n01639765',partNum,root_tar);
[normDist(4,:,:),results(4,:,:),loc(4,:,:)]=do_baseline_part('n01662784',partNum,root_tar);
[normDist(5,:,:),results(5,:,:),loc(5,:,:)]=do_baseline_part('n01674464',partNum,root_tar);
[normDist(6,:,:),results(6,:,:),loc(6,:,:)]=do_baseline_part('n01882714',partNum,root_tar);
[normDist(7,:,:),results(7,:,:),loc(7,:,:)]=do_baseline_part('n01982650',partNum,root_tar);
[normDist(8,:,:),results(8,:,:),loc(8,:,:)]=do_baseline_part('n02084071',partNum,root_tar);
[normDist(9,:,:),results(9,:,:),loc(9,:,:)]=do_baseline_part('n02118333',partNum,root_tar);
[normDist(10,:,:),results(10,:,:),loc(10,:,:)]=do_baseline_part('n02121808',partNum,root_tar);
[normDist(11,:,:),results(11,:,:),loc(11,:,:)]=do_baseline_part('n02129165',partNum,root_tar);
[normDist(12,:,:),results(12,:,:),loc(12,:,:)]=do_baseline_part('n02129604',partNum,root_tar);
[normDist(13,:,:),results(13,:,:),loc(13,:,:)]=do_baseline_part('n02131653',partNum,root_tar);
[normDist(14,:,:),results(14,:,:),loc(14,:,:)]=do_baseline_part('n02324045',partNum,root_tar);
[normDist(15,:,:),results(15,:,:),loc(15,:,:)]=do_baseline_part('n02342885',partNum,root_tar);
[normDist(16,:,:),results(16,:,:),loc(16,:,:)]=do_baseline_part('n02355227',partNum,root_tar);
[normDist(17,:,:),results(17,:,:),loc(17,:,:)]=do_baseline_part('n02374451',partNum,root_tar);
[normDist(18,:,:),results(18,:,:),loc(18,:,:)]=do_baseline_part('n02391049',partNum,root_tar);
[normDist(19,:,:),results(19,:,:),loc(19,:,:)]=do_baseline_part('n02395003',partNum,root_tar);
[normDist(20,:,:),results(20,:,:),loc(20,:,:)]=do_baseline_part('n02398521',partNum,root_tar);
[normDist(21,:,:),results(21,:,:),loc(21,:,:)]=do_baseline_part('n02402425',partNum,root_tar);
[normDist(22,:,:),results(22,:,:),loc(22,:,:)]=do_baseline_part('n02411705',partNum,root_tar);
[normDist(23,:,:),results(23,:,:),loc(23,:,:)]=do_baseline_part('n02419796',partNum,root_tar);
[normDist(24,:,:),results(24,:,:),loc(24,:,:)]=do_baseline_part('n02437136',partNum,root_tar);
[normDist(25,:,:),results(25,:,:),loc(25,:,:)]=do_baseline_part('n02444819',partNum,root_tar);
[normDist(26,:,:),results(26,:,:),loc(26,:,:)]=do_baseline_part('n02454379',partNum,root_tar);
[normDist(27,:,:),results(27,:,:),loc(27,:,:)]=do_baseline_part('n02484322',partNum,root_tar);
[normDist(28,:,:),results(28,:,:),loc(28,:,:)]=do_baseline_part('n02503517',partNum,root_tar);
[normDist(29,:,:),results(29,:,:),loc(29,:,:)]=do_baseline_part('n02509815',partNum,root_tar);
[normDist(30,:,:),results(30,:,:),loc(30,:,:)]=do_baseline_part('n02510455',partNum,root_tar);
disp(results);
disp(normDist);
disp(loc);
end


function [normDist,result,loc]=do_baseline_part(Name_batch,partNum,root_tar)
result=zeros(1,partNum);
normDist=zeros(1,partNum);
loc=zeros(1,partNum);
Name_batch_target=[root_tar,Name_batch];
for i=1:partNum
    try
        [result(i),normDist(i),loc(i)]=step_getResult(Name_batch,i,Name_batch_target);
    catch
        fprintf('Cannot get results for %s  part %02d\n',Name_batch_target,i)
    end
end
fprintf('%s   meanResult %f\n',Name_batch,mean(result))
end

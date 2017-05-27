function xHW=List2xHW_(list,h)
xHW=zeros(2,length(list));
xHW(1,:)=mod(list'-1,h)+1;
xHW(2,:)=(list'-xHW(1,:))./h+1;
end

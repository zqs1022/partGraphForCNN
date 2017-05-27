/*
for i=1:listLen
    DepID=part.parent(i).DepID;
    ThePHW=part.parent(i).DeltaHW;
    for j=1:parentNum
        Dep=DepID(1,j);
        ID=DepID(2,j);
        if(parent_record(Dep).valid(ID)==true)
            parent.pHW(:,j,i)=parent_record(Dep).pHW(:,ID)+ThePHW(:,j);
            parent.valid(j,i)=true;
        end
    end
end
*/

#include<iostream>
#include "mex.h"
#include "matrix.h"
#include "stdio.h"
#include "math.h"
#include<string.h>

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
	if(nrhs!=3)
		mexErrMsgTxt("\nErrors in input.\n");
	
    const char* fname;
    int ifield,nfields,num_parent,num_record,i,j,Dep,ID,parentNum;
    mxArray *DepID,*DeltaHW;
    double *DepID_d,*DeltaHW_d,*valid_d,*pHW_d,DeltaH,DeltaW,*pHW_record,*pHW_out,*valid_out,valid;

    //readinfo
    nfields=mxGetNumberOfFields(prhs[0]);
    num_parent=mxGetNumberOfElements(prhs[0]);
    if((nfields!=2)||(strcmp(mxGetFieldNameByNumber(prhs[0],0),"DepID")!=0)||(strcmp(mxGetFieldNameByNumber(prhs[0],1),"DeltaHW")!=0)){
        mexErrMsgTxt("\nErrors in input.\n");
    }
    nfields=mxGetNumberOfFields(prhs[0]);
    num_record=mxGetNumberOfElements(prhs[0]);
    if((nfields!=2)||(strcmp(mxGetFieldNameByNumber(prhs[1],0),"valid")!=0)||(strcmp(mxGetFieldNameByNumber(prhs[1],1),"pHW")!=0))
        mexErrMsgTxt("\nErrors in input.\n");
    parentNum=int(*((double*)mxGetPr(prhs[2])));
    
    //processing
    plhs[0]=mxCreateDoubleMatrix(2*parentNum,num_parent,mxREAL);
	pHW_out=mxGetPr(plhs[0]);
	plhs[1]=mxCreateDoubleMatrix(parentNum,num_parent,mxREAL);
	valid_out=mxGetPr(plhs[1]);
    memset(pHW_out,0,sizeof(mxREAL)*2*parentNum*num_parent);
    memset(valid_out,0,sizeof(mxREAL)*parentNum*num_parent);
    
    for(i=0;i<num_parent;i++){
        DepID=mxGetFieldByNumber(prhs[0],i,0);
        DeltaHW=mxGetFieldByNumber(prhs[0],i,1);
        if((mxGetNumberOfElements(DepID)!=parentNum*2)||(mxGetNumberOfElements(DeltaHW)!=parentNum*2))
            mexErrMsgTxt("\nErrors in input.\n");
        DepID_d=((double*)mxGetData(DepID));
        DeltaHW_d=((double*)mxGetData(DeltaHW));
        for(j=0;j<parentNum;j++){
            Dep=int(DepID_d[j*2]);
            ID=int(DepID_d[j*2+1]);
            DeltaH=DeltaHW_d[j*2];
            DeltaW=DeltaHW_d[j*2+1];
            valid=((double*)mxGetData(mxGetFieldByNumber(prhs[1],Dep-1,0)))[ID-1];
            pHW_record=((double*)mxGetData(mxGetFieldByNumber(prhs[1],Dep-1,1)));
            if(valid==1){
                pHW_out[(i*parentNum+j)*2]=pHW_record[(ID-1)*2]+DeltaH;
                pHW_out[(i*parentNum+j)*2+1]=pHW_record[(ID-1)*2+1]+DeltaW;
                valid_out[i*parentNum+j]=1;
            }
        }
    }
}

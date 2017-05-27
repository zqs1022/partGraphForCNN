#include<iostream>
#include"string.h"
#include "mex.h"
#include "matrix.h"
#include "stdio.h"
#include "math.h"

#define sqr(a) ((a)*(a))

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
	if(nrhs!=3)
		mexErrMsgTxt("\nErrors in input.\n");
	
    double *fea,*fea_flip,*prob,*cosdist,*cosdist_flip,*norm,*norm_flip,theP;
    int objNum,dimNum,idx,idx_1,idx_2,i,o1,o2;
    
    //readinfo
    prob=((double*)mxGetPr(prhs[0]));
    fea=((double*)mxGetPr(prhs[1]));
    fea_flip=((double*)mxGetPr(prhs[2]));
    dimNum=mxGetN(prhs[0]);
    objNum=mxGetN(prhs[1]);
    if((dimNum!=mxGetM(prhs[1]))||(dimNum!=mxGetM(prhs[2]))||(objNum!=mxGetN(prhs[2])))
        mexErrMsgTxt("\nErrors in input.\n");
    
    //output memory
	plhs[0]=mxCreateDoubleMatrix(objNum,objNum,mxREAL);
    cosdist=mxGetPr(plhs[0]);
    plhs[1]=mxCreateDoubleMatrix(objNum,objNum,mxREAL);
    cosdist_flip=mxGetPr(plhs[1]);
    plhs[2]=mxCreateDoubleMatrix(objNum,1,mxREAL);
	norm=mxGetPr(plhs[2]);
    plhs[3]=mxCreateDoubleMatrix(objNum,1,mxREAL);
	norm_flip=mxGetPr(plhs[3]);

    //processing
    memset(cosdist,0,sizeof(double)*sqr(objNum));
    memset(cosdist_flip,0,sizeof(double)*sqr(objNum));
    memset(norm,0,sizeof(double)*objNum);
    memset(norm_flip,0,sizeof(double)*objNum);
    for(i=0;i<dimNum;i++){
        theP=prob[i];
        for(o1=0;o1<objNum;o1++){
            idx_1=i+o1*dimNum;
            //mexPrintf("%d %d\n",idx,idx_1);
            norm[o1]+=sqr(fea[idx_1])*theP;
            norm_flip[o1]+=sqr(fea_flip[idx_1])*theP;
            for(o2=0;o2<objNum;o2++){
                idx_2=i+o2*dimNum;
                idx=o1+o2*objNum;
                //mexPrintf("%d %d %d\n",idx,idx_1,idx_2);
                cosdist[idx]+=fea[idx_2]*fea[idx_1]*theP;
                cosdist_flip[idx]+=fea[idx_2]*fea_flip[idx_1]*theP;
            }
        }
    }
}

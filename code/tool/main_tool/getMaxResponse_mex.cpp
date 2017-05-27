#include<iostream>
#include "mex.h"
#include "matrix.h"
#include "stdio.h"
#include "math.h"
#include<string.h>

#define sqr(a) ((a)*(a))


inline void check(int a,int mapSize){
    if((a<=0)||(a>mapSize))
        mexErrMsgTxt("\nErrors in input: input coordinates were not in the map.\n");
}


double getValue(double* pHW,int h,int w,int latNum){
    double v;
    int i;
    v=0;
    for(i=0;i<latNum;i++){
        v+=-(sqr(h-pHW[i*2])+sqr(w-pHW[i*2+1]))/2.0;
    }
    v/=latNum;
    return v;
}


void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
	if(nrhs!=11)
		mexErrMsgTxt("\nErrors in input #1.\n");
	
    double *h_min,*h_max,*w_min,*w_max,*pHW,*pHW_c1,*pHW_c2,*pHW_c3,*pHW_c4;
    double h_min_tar,h_max_tar,w_min_tar,w_max_tar,value_tar,value_tmp,v,v1,v2,v3,v4,centerWeight;
    double *o1,*o2,*o3,*o4,*o5;
    int choiceNum,mapSize,latNum,latNum_c1,latNum_c2,latNum_c3,latNum_c4,hMin,hMax,wMin,wMax,hC,wC,i,j,k,l;
    
    //readinfo
    h_min=(double*)mxGetPr(prhs[0]);
    w_min=(double*)mxGetPr(prhs[1]);
    h_max=(double*)mxGetPr(prhs[2]);
    w_max=(double*)mxGetPr(prhs[3]);
    pHW=(double*)mxGetPr(prhs[4]);
    pHW_c1=(double*)mxGetPr(prhs[5]);
    pHW_c2=(double*)mxGetPr(prhs[6]);
    pHW_c3=(double*)mxGetPr(prhs[7]);
    pHW_c4=(double*)mxGetPr(prhs[8]);
    centerWeight=*((double*)mxGetPr(prhs[9]));
    mapSize=(int(*((double*)mxGetPr(prhs[10]))));
    choiceNum=mxGetM(prhs[0])*mxGetN(prhs[0]);
    for(i=1;i<4;i++){
        if(mxGetM(prhs[i])*mxGetN(prhs[i])!=choiceNum)
            mexErrMsgTxt("\nErrors in input #2.\n");
    }
    latNum=mxGetN(prhs[4]);
    latNum_c1=mxGetN(prhs[5]);
    latNum_c2=mxGetN(prhs[6]);
    latNum_c3=mxGetN(prhs[7]);
    latNum_c4=mxGetN(prhs[8]);
    
    //processing
    value_tar=-1000000000000;
    for(i=0;i<choiceNum;i++){
        hMin=round(h_min[i]);
        check(hMin,mapSize);
        for(j=0;j<choiceNum;j++){
            hMax=round(h_max[j]);
            check(hMax,mapSize);
            hC=round((h_min[i]+h_max[j])/2.0);
            for(k=0;k<choiceNum;k++){
                wMin=round(w_min[k]);
                check(wMin,mapSize);
                v1=getValue(pHW_c1,hMin,wMin,latNum_c1);
                v2=getValue(pHW_c2,hMax,wMin,latNum_c2);
                for(l=0;l<choiceNum;l++){
                    wMax=round(w_max[l]);
                    check(wMax,mapSize);
                    wC=round((w_min[k]+w_max[l])/2.0);
                    v=getValue(pHW,hC,wC,latNum);
                    v3=getValue(pHW_c3,hMin,wMax,latNum_c3);
                    v4=getValue(pHW_c4,hMax,wMax,latNum_c4);
                    value_tmp=(v*centerWeight+v1+v2+v3+v4)/(centerWeight+4);
                    if(value_tmp>value_tar){
                        value_tar=value_tmp;
                        h_min_tar=hMin;
                        h_max_tar=hMax;
                        w_min_tar=wMin;
                        w_max_tar=wMax;
                    }
                }
            }
        }
    }
    plhs[0]=mxCreateDoubleMatrix(1,1,mxREAL);
    plhs[1]=mxCreateDoubleMatrix(1,1,mxREAL);
    plhs[2]=mxCreateDoubleMatrix(1,1,mxREAL);
    plhs[3]=mxCreateDoubleMatrix(1,1,mxREAL);
    plhs[4]=mxCreateDoubleMatrix(1,1,mxREAL);
	o1=mxGetPr(plhs[0]);
    o2=mxGetPr(plhs[1]);
    o3=mxGetPr(plhs[2]);
    o4=mxGetPr(plhs[3]);
    o5=mxGetPr(plhs[4]);
    *o1=h_min_tar;
    *o2=w_min_tar;
    *o3=h_max_tar;
    *o4=w_max_tar;
    *o5=value_tar;
}

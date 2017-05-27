#include<iostream>
#include "mex.h"
#include "matrix.h"
#include "stdio.h"
#include "math.h"

#define sqr(a) ((a)*(a))
#define max(a,b) ((a)>(b)?(a):(b))

void x2p(double xH,double xW,int Stride,double centerStart,double* pH,double* pW){
    *pH=centerStart+(xH-1)*Stride;
    *pW=centerStart+(xW-1)*Stride;
}


inline double getLogGauss(double mu1,double mu2, double* rho){
    return -(sqr(mu1)+sqr(mu2))/max((2*(sqr(rho[0])+sqr(rho[1]))),0.00000000001);
}


inline int getLayIndex(double* xHW,double* xHxW, double depth){
    return int(xHW[0]+(xHW[1]-1)*xHxW[0]+(depth-1)*xHxW[0]*xHxW[1])-1;
}


void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){
	if(nrhs!=22)
		mexErrMsgTxt("\nErrors in input.\n");
	
    double *init_xHW,*x,*valid,*depth,*part_DeltaHW,*pHW_scale,*pHW_center,*label_center,*xHxW,invalidX,minValue,*fset,*partHOG,centerStart,*gen_local,*gen_geo,*gen_SUM,*parent_pHW,*parent_valid,*weight;
    int num,SearchStride,Stride,i,j,k,dh,dw,parentNum,idx,idx0,HOGDim;
    bool hasLabel;
    double w_local,w_geo,w_app,w_reli,w_gen,w_parent,w_super_loc,w_delta,pH,pW,init_pH,init_pW;
    double delta[2],tmp_xHW[2],init_xH,init_xW,tmp_local,tmp_geo,tmp_pHW_part[2],tmp_parent,tmp_app,mu1,mu2,tmp_super_loc,tmp_GenL,tmp_DisL,tmp_GenDisL;
    double (*pHW_part)[2],(*v_xHW)[2],*v_local,*v_geo,*v_app,*v_gen_local,*v_gen_geo,*v_gen_SUM,*v_parent,*v_super_loc,*v_reli,*v_GenL,*v_DisL,*v_GenDisL;
    double *f0,*f1,*f2,*f3,*f4,*f5,*f6,*f7,*f8,*f9,*f10,*f11,*f12,*f13;
    
    //readinfo
    init_xHW=((double*)mxGetPr(prhs[0]));
    num=mxGetN(prhs[0]);
    x=((double*)mxGetPr(prhs[1]));
    valid=((double*)mxGetPr(prhs[2]));
    depth=((double*)mxGetPr(prhs[3]));
    part_DeltaHW=((double*)mxGetPr(prhs[4]));
    pHW_scale=((double*)mxGetPr(prhs[5]));
    pHW_center=((double*)mxGetPr(prhs[6]));
    label_center=((double*)mxGetPr(prhs[7]));
    hasLabel=(mxGetM(prhs[7])*mxGetN(prhs[7])>0);
    xHxW=((double*)mxGetPr(prhs[8]));
    invalidX=*((double*)mxGetPr(prhs[9]));
    minValue=*((double*)mxGetPr(prhs[10]));
    fset=((double*)mxGetPr(prhs[11]));
    HOGDim=mxGetM(prhs[11]);
    SearchStride=int(*((double*)mxGetPr(prhs[12])));
    Stride=int(*((double*)mxGetPr(prhs[13])));
    centerStart=*((double*)mxGetPr(prhs[14]));
    gen_local=((double*)mxGetPr(prhs[15]));
    gen_geo=((double*)mxGetPr(prhs[16]));
    gen_SUM=((double*)mxGetPr(prhs[17]));
    parent_pHW=((double*)mxGetPr(prhs[18]));
    parent_valid=((double*)mxGetPr(prhs[19]));
    parentNum=mxGetM(prhs[19]);
    weight=((double*)mxGetPr(prhs[20]));
    partHOG=((double*)mxGetPr(prhs[21]));
    w_local=weight[0];
    w_geo=weight[1];
    w_app=weight[2];
    w_reli=weight[3];
    w_gen=weight[4];
    w_parent=weight[5];
    w_super_loc=weight[6];
    w_delta=weight[7];
    
    //malloc memory
    pHW_part=v_xHW=NULL;
    v_local=v_geo=v_app=v_gen_local=v_gen_geo=v_gen_SUM=v_parent=v_super_loc=v_reli=v_GenL=v_DisL=v_GenDisL=NULL;
    while(pHW_part==NULL) pHW_part=new(std::nothrow) double[num][2];
    while(v_xHW==NULL) v_xHW=new(std::nothrow) double[num][2];
    while(v_local==NULL) v_local=new(std::nothrow) double[num];
    while(v_geo==NULL) v_geo=new(std::nothrow) double[num];
    while(v_app==NULL) v_app=new(std::nothrow) double[num];
    v_gen_local=gen_local; //
    v_gen_geo=gen_geo; //
    v_gen_SUM=gen_SUM; //
    while(v_parent==NULL) v_parent=new(std::nothrow) double[num];
    while(v_super_loc==NULL) v_super_loc=new(std::nothrow) double[num];
    while(v_reli==NULL) v_reli=new(std::nothrow) double[num]; //
    while(v_GenL==NULL) v_GenL=new(std::nothrow) double[num];
    while(v_DisL==NULL) v_DisL=new(std::nothrow) double[num];
    while(v_GenDisL==NULL) v_GenDisL=new(std::nothrow) double[num];
    //processing
    //delta[0]=sqrt(sqr(Stride)+sqr(pHW_scale[0]));
    //delta[1]=sqrt(sqr(Stride)+sqr(pHW_scale[1]));
    delta[0]=w_delta;
    delta[1]=w_delta;
    for(i=0;i<num;i++){
        init_xH=init_xHW[i*2];
        init_xW=init_xHW[i*2+1];
        x2p(init_xH,init_xW,Stride,centerStart,&init_pH,&init_pW);
        //v_reli[i]=getLogGauss(init_pH-pHW_center[0],init_pW-pHW_center[1],pHW_scale);
        v_reli[i]=getLogGauss(init_pH-pHW_center[0],init_pW-pHW_center[1],delta);
        v_GenL[i]=minValue;
        for(dw=-SearchStride;dw<=SearchStride;dw++){
            for(dh=-SearchStride;dh<=SearchStride;dh++){
                if((init_xH+dh>0)&&(init_xH+dh<=xHxW[0])&&(init_xW+dw>0)&&(init_xW+dw<=xHxW[1])){
                    tmp_xHW[0]=init_xH+dh;
                    tmp_xHW[1]=init_xW+dw;
                    tmp_app=0;
                    idx0=getLayIndex(tmp_xHW,xHxW,1);
                    for(k=0;k<HOGDim;k++) tmp_app+=-sqr(partHOG[k+i*HOGDim]-fset[k+idx0*HOGDim]);
                    idx=getLayIndex(tmp_xHW,xHxW,depth[i]);
                    if(valid[idx]==0)
                        tmp_local=invalidX;
                    else
                        tmp_local=x[idx];
                    tmp_geo=getLogGauss(dh*Stride,dw*Stride,delta);
                    x2p(tmp_xHW[0],tmp_xHW[1],Stride,centerStart,&pH,&pW);
                    tmp_pHW_part[0]=pH+part_DeltaHW[i*2];
                    tmp_pHW_part[1]=pW+part_DeltaHW[i*2+1];
                    if(parentNum>0){
                        tmp_parent=0;
                        for(j=0;j<parentNum;j++){
                            mu1=pH-parent_pHW[j*2+i*(2*parentNum)];
                            mu2=pW-parent_pHW[1+j*2+i*(2*parentNum)];
                            if(parent_valid[j+i*parentNum]){
                                //tmp_parent+=exp(-sqrt(sqr(mu1)+sqr(mu2))/(sqrt(sqr(pHW_scale[0])+sqr(pHW_scale[1]))/2))/parentNum;
                                tmp_parent+=-(sqrt(sqr(mu1)+sqr(mu2))/(sqrt(sqr(delta[0])+sqr(delta[1]))/2))/parentNum;
                                //tmp_parent+=-sqr(sqrt(sqr(mu1)+sqr(mu2))/(sqrt(sqr(pHW_scale[0])+sqr(pHW_scale[1]))/2))/parentNum;
                            }
                        }
                    }
                    else
                        tmp_parent=0;
                    if(hasLabel){
                        mu1=(label_center[0]-tmp_pHW_part[0])*2;
                        mu2=(label_center[1]-tmp_pHW_part[1])*2;
                        //tmp_super_loc=getLogGauss(mu1,mu2,pHW_scale);
                        tmp_super_loc=getLogGauss(mu1,mu2,delta);
                    }
                    else
                        tmp_super_loc=0;
                    tmp_GenL=tmp_local*w_local+tmp_geo*w_geo+tmp_app*w_app+v_gen_SUM[i]*w_gen+tmp_parent*w_parent;
                    /*if((dh==0)&&(dw==0)){
                        mexPrintf("%lf score. i=%d\n %lf %lf %lf %lf %lf\n",tmp_GenL,i,tmp_local,tmp_geo,tmp_app,v_gen_SUM,tmp_parent);
                    }*/
                    if((tmp_GenL>v_GenL[i])||(v_GenL[i]==minValue)){
                        v_GenL[i]=tmp_GenL;
                        v_DisL[i]=tmp_super_loc*w_super_loc;
                        v_GenDisL[i]=v_GenL[i]+v_DisL[i]+v_reli[i]*w_reli;
                        for(j=0;j<2;j++){
                            pHW_part[i][j]=tmp_pHW_part[j];
                            v_xHW[i][j]=tmp_xHW[j];
                        }
                        v_local[i]=tmp_local;
                        v_geo[i]=tmp_geo;
                        v_parent[i]=tmp_parent;
                        v_super_loc[i]=tmp_super_loc;
                        v_app[i]=tmp_app;
                    }
                }
                /*if(i==0)
                    mexPrintf("abcde (%d,%d)  %lf\n",dh,dw,v_GenL[0]);*/
            }
        }
    }
    
    //output
	plhs[0]=mxCreateDoubleMatrix(2,num,mxREAL);
	f0=mxGetPr(plhs[0]);
	plhs[1]=mxCreateDoubleMatrix(2,num,mxREAL);
	f1=mxGetPr(plhs[1]);
	plhs[2]=mxCreateDoubleMatrix(num,1,mxREAL);
	f2=mxGetPr(plhs[2]);
	plhs[3]=mxCreateDoubleMatrix(num,1,mxREAL);
	f3=mxGetPr(plhs[3]);
	plhs[4]=mxCreateDoubleMatrix(num,1,mxREAL);
	f4=mxGetPr(plhs[4]);
	plhs[5]=mxCreateDoubleMatrix(num,1,mxREAL);
	f5=mxGetPr(plhs[5]);
	plhs[6]=mxCreateDoubleMatrix(num,1,mxREAL);
	f6=mxGetPr(plhs[6]);
	plhs[7]=mxCreateDoubleMatrix(num,1,mxREAL);
	f7=mxGetPr(plhs[7]);
	plhs[8]=mxCreateDoubleMatrix(num,1,mxREAL);
	f8=mxGetPr(plhs[8]);
    plhs[9]=mxCreateDoubleMatrix(num,1,mxREAL);
	f9=mxGetPr(plhs[9]);
	plhs[10]=mxCreateDoubleMatrix(num,1,mxREAL);
	f10=mxGetPr(plhs[10]);
	plhs[11]=mxCreateDoubleMatrix(num,1,mxREAL);
	f11=mxGetPr(plhs[11]);
	plhs[12]=mxCreateDoubleMatrix(num,1,mxREAL);
	f12=mxGetPr(plhs[12]);
	plhs[13]=mxCreateDoubleMatrix(num,1,mxREAL);
	f13=mxGetPr(plhs[13]);
    for(i=0;i<num;i++){
        for(j=0;j<2;j++){
            f0[i*2+j]=pHW_part[i][j];
            f1[i*2+j]=v_xHW[i][j];
        }
        f2[i]=v_local[i];
        f3[i]=v_geo[i];
        f4[i]=v_app[i];
        f5[i]=v_gen_local[i];
        f6[i]=v_gen_geo[i];
        f7[i]=v_gen_SUM[i];
        f8[i]=v_parent[i];
        f9[i]=v_super_loc[i];
        f10[i]=v_reli[i];
        f11[i]=v_GenL[i];
        f12[i]=v_DisL[i];
        f13[i]=v_GenDisL[i];
    }
    //mexPrintf("abcde   %lf\n",v_GenL[0]);
    delete[] pHW_part;
    delete[] v_xHW;
    delete[] v_local;
    delete[] v_geo;
    delete[] v_parent;
    delete[] v_super_loc;
    delete[] v_reli;
    delete[] v_GenL;
    delete[] v_DisL;
    delete[] v_GenDisL;
}

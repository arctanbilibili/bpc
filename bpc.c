#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "martix.h"

#define ss 0.2
typedef struct{
	int cells;//本层
	int cells_b;//上一层神经元

	int bn;//0关1开
	double mean;
	double std;
	MARTIX *xif;//bn层输入//xiforward

	MARTIX *xi;
	MARTIX *xo;
	MARTIX *b;
	MARTIX *loss;

	MARTIX *tmploss;//
	MARTIX *wnn;//二维拉成一维
}BPLINE;

typedef struct{
	int floors;
	BPLINE* line;
}NET;

//tanh
double tanh(double x)
{
	double e1=exp(x), e2=exp(-x);
	return (e1-e2)/(e1+e2);
}
double tanhxo(double xo)
{
	return 1-xo*xo;
}
//调用函数
double f(double x)
{
	return tanh(x);
}
double f1xo(double xo)//微分的另一种形式 
{
	return tanhxo(xo);
}

NET* netCreat(int *stru,int floors){
	//int floors = sizeof(stru)/sizeof(stru[0]);
	//printf("%d\r\n",floors);
	NET* net = (NET*)malloc(sizeof(NET));
	net->line = (BPLINE*)malloc(floors*sizeof(BPLINE));

	net->floors = floors;
//输入层

	net->line[0].cells = stru[0];
	net->line[0].cells_b = 0;
	net->line[0].xif = MARTIX_cre(stru[0],1);
	net->line[0].xi = MARTIX_cre(stru[0],1);
	net->line[0].xo = MARTIX_cre(stru[0],1);
	net->line[0].b  = MARTIX_cre(stru[0],1);
	net->line[0].loss = MARTIX_cre(stru[0],1);
	net->line[0].tmploss = MARTIX_cre(stru[0],1);
	net->line[0].wnn=NULL;
//隐层+输出层
	for(int floor=1;floor<floors;floor++){
		net->line[floor].cells = stru[floor];
		net->line[floor].cells_b = stru[floor-1];
		net->line[floor].xif = MARTIX_cre(stru[floor],1);
		net->line[floor].xi = MARTIX_cre(stru[floor],1);
		net->line[floor].xo = MARTIX_cre(stru[floor],1);
		net->line[floor].b  = MARTIX_cre(stru[floor],1);
		net->line[floor].loss = MARTIX_cre(stru[floor],1);
		net->line[floor].tmploss = MARTIX_cre(stru[floor],1);
		net->line[floor].wnn = MARTIX_cre(stru[floor],stru[floor-1]);
	}
	return net;
}
//rand()/65536.0
void netInit(NET* net){
	//初始化所有基础参数
	for(int floor=0;floor<net->floors;floor++){
		for(int i=0;i<net->line[floor].cells;i++){
			net->line[floor].xi->val[i]=rand()/65536.0;
			net->line[floor].xo->val[i]=rand()/65536.0;
			net->line[floor].b ->val[i]=rand()/65536.0;
		}
	}
	//初始化1-n层wnn
	for(int floor=1;floor<net->floors;floor++){
		for(int i=0;i<net->line[floor].cells;i++){
			int col_b = net->line[floor].cells_b;
			for(int j=0;j<net->line[floor].cells_b;j++){
				net->line[floor].wnn->val[i*col_b+j]=rand()/65536.0;
			}
		}
	}
}

void netSetBn(NET* net,int* bn){
	for(int floor=0;floor<net->floors;floor++){
		net->line[floor].bn = bn[floor];
	}
}

void netDel(NET* net){
	for(int floor=0;floor<net->floors;floor++){
		BPLINE line = net->line[floor];
		MARTIX_del(line.xif);
		MARTIX_del(line.xi);
		MARTIX_del(line.xo);
		MARTIX_del(line.b);
		MARTIX_del(line.loss);
		MARTIX_del(line.tmploss);
		MARTIX_del(line.wnn);
	}
	free(net->line);
	free(net);
}

double calMean(double*val,int len){
	double tmp=0;
	for(int i=0;i<len;i++){
		tmp+=val[i];
	}
	return tmp/len;
}

double calStd(double*val,int len,double mean)//标准差
{
	double sum=0;
	for (int i = 0; i < len;i++){
		sum += (val[i]-mean)*(val[i]-mean);
	}
	return sqrt(sum / len);
}

void forward(NET* net,double*xtrain){
	BPLINE line;
	BPLINE line_b;
	MARTIX* tmp;

	for(int i=0;i<net->line[0].xi->rol;i++){//xif赋值
		net->line[0].xif->val[i] = xtrain[i];
	}
	line = net->line[0];//第0行

	//start:
	net->line[0].mean = line.mean = calMean(line.xif->val,line.cells);
	net->line[0].std = line.std = calStd(line.xif->val,line.cells,line.mean);
	if(line.bn==1){
		tmp = MARTIX_copy(line.xif);
		MARTIX_add_para_cover(tmp,line.xif,-line.mean);
		MARTIX_mul_para_cover(tmp,tmp,1/line.std);
		MARTIX_ncover(line.xi,tmp);

		MARTIX_add_cover(tmp,line.xi,line.b);//tmp留用
	}
	else{
		MARTIX_ncover(line.xi,line.xif);
		tmp = MARTIX_add(line.xi,line.b);//tmp留用
	}
	
	MARTIX_f(line.xo,tmp,f);
	MARTIX_del(tmp);

	for(int floor=1;floor < net->floors;floor++){//第floor行4
		line = net->line[floor];
		line_b = net->line[floor-1];

		MARTIX_mul_cover(line.xif,line.wnn,line_b.xo);//计算xif

		//start:
		net->line[floor].mean = line.mean = calMean(line.xif->val,line.cells);
		net->line[floor].std = line.std = calStd(line.xif->val,line.cells,line.mean);
	// printf("%lf\r\n",line.mean);
	// printf("%lf\r\n",line.std);
	// getchar();
		if(line.bn==1){
			tmp = MARTIX_copy(line.xif);
			MARTIX_add_para_cover(tmp,line.xif,-line.mean);
			MARTIX_mul_para_cover(tmp,tmp,1/line.std);
			MARTIX_ncover(line.xi,tmp);

			MARTIX_add_cover(tmp,line.xi,line.b);//tmp留用
		}
		else{
			MARTIX_ncover(line.xi,line.xif);
			tmp = MARTIX_add(line.xi,line.b);//tmp留用
		}

		MARTIX_f(line.xo,tmp,f);
		MARTIX_del(tmp);
	}
}

void backward(NET* net,double*ytrain){
	BPLINE line;
	BPLINE line_b;

	line = net->line[net->floors-1];//输出层
	MARTIX* ym = MARTIX_cre(line.cells,1);
	for(int i=0;i<line.cells;i++){//ytrain赋值
		ym->val[i] = ytrain[i];
	}	
	MARTIX_sub_cover(line.loss,line.xo,ym);
	MARTIX_del(ym);

	for(int floor=net->floors-1;floor>0;floor--){
		line   = net->line[floor];
		line_b = net->line[floor-1];
		MARTIX* tmp    = MARTIX_copy(line.xo);
		MARTIX* tmpb   = MARTIX_copy(line.b);
		MARTIX* tmploss= MARTIX_copy(line.loss);
		MARTIX* tmpwnn = MARTIX_copy(line.wnn);
		MARTIX* tmpbxoT   = MARTIX_T(line_b.xo);
		MARTIX* tmpT   = MARTIX_T(line.wnn);//计算上一层loss的临时变量

		MARTIX_f(tmp,line.xo,f1xo);
		MARTIX_mul_dot_cover(tmp,tmp,line.loss);

		//tmp == dloss/dxi == dloss/db
		MARTIX_mul_para_cover(tmpb,tmp,ss);//*ss基底
		MARTIX_sub_cover(line.b,line.b,tmpb);//训练b

		//bn修正tmp，和给上层用的tmploss
		if(line.bn==1){
			double tmp1 = MARTIX_sum(tmp);
			MARTIX* a = MARTIX_cre_paras(tmp->rol,tmp->col,tmp1);
			MARTIX* b = MARTIX_copy(line.xi);
			MARTIX* c = MARTIX_copy(tmp);
			MARTIX_mul_dot_cover(c,tmp,line.xi);
			double tmp2 = MARTIX_sum(c);

			MARTIX_mul_para_cover(b,b,tmp2);
			MARTIX_sub_cover(a,a,b);
			MARTIX_mul_para_cover(a,a,1.0/line.cells/line.std);
			MARTIX_mul_para_cover(tmp,tmp,1.0/line.std);
			MARTIX_add_cover(tmp,tmp,a);

			MARTIX_del(a);
			MARTIX_del(b);
			MARTIX_del(c);
		}
		else{
			//tmp不变 dloss/dxif=dloss/dxi
		}
		MARTIX_mul_cover(tmpwnn,tmp,tmpbxoT);
		MARTIX_mul_para_cover(tmpwnn,tmpwnn,ss);//*ss基底

		MARTIX_sub_cover(line.wnn,line.wnn,tmpwnn);//训练wnn

		MARTIX_ncover(tmploss,tmp);//dloss/dxif
		MARTIX_mul_cover(line_b.loss,tmpT,tmploss);
		MARTIX_ncover(line.tmploss,tmploss);

		MARTIX_del(tmp);
		MARTIX_del(tmpb);
		MARTIX_del(tmpwnn);
		MARTIX_del(tmpbxoT);
		MARTIX_del(tmpT);
		MARTIX_del(tmploss);
	}

}

void showNet(NET* net){
	for(int floor=0;floor<net->floors;floor++){
		printf(GREEN"Floor %d : \r\n"NONE,floor);
		showMat(net->line[floor].xi,"xi");
		showMat(net->line[floor].xo,"xo");
		showMat(net->line[floor].b,"b");
		showMat(net->line[floor].wnn,"wnn");
	}
}

void saveModel(char* fname,NET* net){
	FILE *fp = NULL;

	fp=fopen(fname,"w+");
	if(fp!=NULL){
		for(int floor=0;floor<net->floors;floor++){
			BPLINE line = net->line[floor];
			char str[20]={0};

			sprintf(str,"%d Line b:",floor);
			showMat_f(line.b,str,fp);

			sprintf(str,"%d Line Wnn:",floor);
			showMat_f(line.wnn,str,fp);

			fprintf(fp,"\n");
		}
		fprintf(fp,"hello");
		fclose(fp);
	}
	else{
		printf("error");
	}
}

void testMat();
void testrand();
void testmain();

#define N 4
void main(){
	//testMat();

	testmain();
	//testrand();

	return ;
}

void testmain(){
	int stru[N] = {5,6,6,5};
	int bnflag[N] = {1,1,1,0};
	char str[10] = {0};

	double x[5] = {-0.8,-0.7,-0.72,-0.66,-0.82};
	double y[5] = {0.5,-0.2,0,0.5,-0.5};

	srand(114514);

	NET* net0 = netCreat(stru,N);
	netInit(net0);
	netSetBn(net0,bnflag);

	for(int n=0;n<20;n++){
		forward(net0,x);
		backward(net0,y);
	}
	showMat(net0->line[N-1].xo,"xo");

	saveModel("model1.txt",net0);

	netDel(net0);
}


void testrand(){
	for(int i=0;i<100;i++){
		printf("%f ",rand()/65536.0);
	}
}
void testMat(){
	MARTIX* A = MARTIX_cre(2,3);
	MARTIX* B = MARTIX_cre(2,3);

	A->val[0]=1;A->val[1]=2;A->val[2]=3;A->val[3]=4;A->val[4]=5;A->val[5]=6;
	B->val[0]=1;B->val[1]=2;B->val[2]=3;B->val[3]=4;B->val[4]=5;B->val[5]=6;
	
	printf("%lf",MARTIX_sum(A));

	showMat(A,"A");
	showMat(B,"B");


}
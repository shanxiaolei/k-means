/**
* 文件:mpi-first.cpp
* 版本: 0.0.0.1
* 作者:单晓磊
* 日期:2017/5/19
* 公司:
* 描述:k-means聚类算法主函数包含MPI部分
*
* 修改记录 :
* ============================================================================
* 序号   修改日期    修改人  修改原因
*  01    2017/5/19    单晓磊  编码规范
*  02    2017/5/26    单晓磊  将mpi部分代码模块化
*  03    2017/6/10    单晓磊  编码规范，加入日志
*  04    2017/9/4   单晓磊  针对增量数据修改，调换了两个函数的位置
*/
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include "kmeans.h"

const double dThreshold=1e-6;
int iNumClusters=0;
int iNumObjs=0;
int iNumCoords=0;
int iLoop_iterations = 0;

int main(int argc,char *argv[])
{
    int iDimNumObjs=0;
    int **iObjects;//[iNumObjs][iNumCoords]
    int *iDimobjects;
    double **dClusters;
    int *iMembership;//[iNumObjs]
    char cFilename[]="bbc.bin";
    double dCurrcost=0.0;
    double dDimCurrcost=0.0;
    double dLastcost=0.0;//用来前后两次代价
    clock_t start,finish;
    double duration;

    int ret;
    int rank,size;
    double starttime,endtime;
    void Mpi_gather(double dDimCurrcost,double &dCurrcost,int *iNewClusterSize,int *&iClustersize,double **dNewClusters_sum,double **&dAll_newClusters_sum);
    bool JudgeEnd(double dCurrcost,double dLastcost);

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    /****************************从文件读入数据进行处理*******************************************/
    starttime=MPI_Wtime();
    CKmeans kmeans;
    if(rank==0){
        File_read(iNumObjs,iNumCoords,iObjects,cFilename);
        kmeans=CKmeans(iNumObjs,iNumCoords,iObjects);
        kmeans.SetInitMode(CKmeans::Initkmeans2);
	iNumClusters=kmeans.Get_k(iNumObjs);
        //测试
        cout<<"the iNumObjs is:"<<iNumObjs<<endl;
        cout<<"the iNumCoords is:"<<iNumCoords<<endl;
	cout<<"the iNumClusters is:"<<iNumClusters<<endl;
        cout<<"the dimsize:"<<iNumObjs/(size-1)<<endl;

    }
    MPI_Bcast(&iNumObjs,1,MPI_INT,0,MPI_COMM_WORLD);  //进程0广播
    MPI_Bcast(&iNumCoords,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&iNumClusters,1,MPI_INT,0,MPI_COMM_WORLD);
    /****************************初始化*******************************************/
    malloc2D(dClusters,iNumClusters,iNumCoords,double);

    if(rank==0)
        {

	    kmeans.SetCluster(iObjects,dClusters);
	}
    else{
        if((iNumObjs%(size-1)!=0)&&rank==size-1)
            iDimNumObjs=iNumObjs-(iNumObjs/(size-1))*(size-2);
        else
            iDimNumObjs=iNumObjs/(size-1);
        iDimobjects=(int*)malloc(iDimNumObjs*iNumCoords*sizeof(int));
    }
    //广播中心点坐标
    for(int i=0;i<iNumClusters;i++)
        MPI_Bcast(dClusters[i],iNumCoords,MPI_DOUBLE,0,MPI_COMM_WORLD);

    //为更新中心点坐标而设
    int *iNewClusterSize; //[iNumClusters]:no.objects assigned in each new cluster
    int *iClustersize;
    double **dNewClusters_sum;//[iNumCoords][iNumClusters]
    double **dAll_newClusters_sum;//[iNumCoords][iNumClusters]
    iNewClusterSize=(int*)malloc(iNumClusters*sizeof(int));
    iClustersize=(int*)malloc(iNumClusters*sizeof(int));
    malloc2D(dNewClusters_sum,iNumClusters,iNumCoords,double);
    malloc2D(dAll_newClusters_sum,iNumClusters,iNumCoords,double);


    /****************************创建远地窗口*******************************************/
    MPI_Win win1;//窗口内存入所有数据点
    MPI_Win win2;
    MPI_Win win3;
    int *iOvership=(int*)malloc(iNumObjs*sizeof(int));
    int *iWinobjs;

    //申请内存空间
	iWinobjs = (int*)malloc(iNumObjs*iNumCoords*sizeof(int));
	iMembership=(int*)malloc(iNumObjs*sizeof(int));
    //创建远地窗口
	ret = MPI_Win_create(iWinobjs, iNumObjs*iNumCoords, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win1);
	if(MPI_SUCCESS != ret)
	{
		cout<<"MPI_Win_create faild[%d],"<<rank<<endl;
		return 1;
	}

    ret = MPI_Win_create(iMembership, iNumObjs, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win2);
	if(MPI_SUCCESS != ret)
	{
		cout<<"MPI_Win_create faild[%d],"<<rank<<endl;
		return 1;
	}

    ret = MPI_Win_create(iOvership, iNumObjs, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win3);
	if(MPI_SUCCESS != ret)
	{
		cout<<"MPI_Win_create faild[%d],"<<rank<<endl;
		return 1;
	}

	/**************************进程0初始化数据，相当于数据发布****************************/
	if(rank==0){
		for(int i=0; i<iNumObjs; i++)
		{
			for(int j=0; j<iNumCoords; j++){
                *(iWinobjs+i*iNumCoords+j) = iObjects[i][j];
			}
            *(iMembership+i)=-1;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);//同步点

	/**************************其他进程读取数据****************************/
    if(rank !=0){

        MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win1);
        ret = MPI_Get(iDimobjects, iDimNumObjs*iNumCoords, MPI_INT, 0, (rank-1)*(iNumObjs/(size-1))*iNumCoords,iDimNumObjs*iNumCoords, MPI_INT, win1);
        MPI_Win_unlock(0,  win1);

    }
/*****************************聚类初始化*****************************************/
    if(rank!=0)
        kmeans=CKmeans(iDimNumObjs,iNumCoords,iNumClusters,iDimobjects);

/*****************************开始进行聚类*****************************************/
do{
    dLastcost=dCurrcost;
    dCurrcost=0;
    dDimCurrcost=0;
    Resetvar(iNumClusters,iNumCoords,iNewClusterSize,dNewClusters_sum,iClustersize);
    if(rank==0)
	start=clock();
    if(rank!=0){
        MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win2);
        ret = MPI_Get(iMembership, iDimNumObjs, MPI_INT, 0,(rank-1)*(iNumObjs/(size-1)),iDimNumObjs, MPI_INT, win2);
        MPI_Win_unlock(0,  win2);
	//cout<<"debug1"<<endl;
        dDimCurrcost=kmeans.Kmeans_cluster(dClusters,iMembership,iNewClusterSize,dNewClusters_sum);
	//cout<<"debug2"<<endl;
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win2);
        ret = MPI_Put(iMembership, iDimNumObjs, MPI_INT, 0,(rank-1)*(iNumObjs/(size-1)),iDimNumObjs, MPI_INT, win2);
        MPI_Win_unlock(0,  win2);
    }
    Mpi_gather(dDimCurrcost,dCurrcost,iNewClusterSize,iClustersize,dNewClusters_sum,dAll_newClusters_sum);

	dCurrcost /= iNumObjs;

    if(rank==0)
    {
        cout<<"the loop:"<<iLoop_iterations<<" Currcost:"<<dCurrcost<<endl;
        kmeans.Updatecenter(dClusters,iClustersize,dAll_newClusters_sum);
    }

    //广播中心点坐标
    for(int i=0;i<iNumClusters;i++)
        MPI_Bcast(dClusters[i],iNumCoords,MPI_DOUBLE,0,MPI_COMM_WORLD);

   
    MPI_Bcast(&dCurrcost,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);//同步点


}while(JudgeEnd(dCurrcost,dLastcost));
	
	if(rank==0){
		finish=clock();
		duration=(double)(finish - start) / CLOCKS_PER_SEC;
    		printf( "%f seconds\n", duration ); 
	}
//测试
    dCurrcost=0;
    dDimCurrcost=0;

    if(rank!=0){
	dDimCurrcost=kmeans.together(dClusters,iMembership);
	}

    MPI_Barrier(MPI_COMM_WORLD);//同步点
    MPI_Reduce(&dDimCurrcost,&dCurrcost,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);  //归约至进程0,进程中每个聚类的数据点与其中心点的距离之和
   if(rank==0)
	cout<<"k:"<<iNumClusters<<" cost:"<<dCurrcost/iNumObjs<<endl;


//the end

    kmeans.Freecuda();

    if(rank==0)
    {
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win3);
        ret = MPI_Put(iMembership, iNumObjs, MPI_INT, 0,0,iNumObjs, MPI_INT, win3);
        MPI_Win_unlock(0,  win3);
        File_write(iNumObjs,iNumCoords,iNumClusters,iOvership,dClusters);
    }
    MPI_Barrier(MPI_COMM_WORLD);//同步点
    MPI_Win_free(&win1);
    MPI_Win_free(&win2);
    MPI_Win_free(&win3);

    if(rank!=0)
        free(iDimobjects);
    endtime=MPI_Wtime();
    if(rank==0)
    {
        cout<<"the whole program's time is "<<(endtime-starttime)<<"s"<<endl;
        free(iObjects[0]);
        free(iObjects);
    }

    MPI_Finalize();

	free(dClusters[0]);
    free(dClusters);
    free(iClustersize);
	free(iNewClusterSize);
	free(dNewClusters_sum[0]);
    free(dNewClusters_sum);
    free(dAll_newClusters_sum[0]);
    free(dAll_newClusters_sum);
	free(iWinobjs);
	free(iMembership);
	free(iOvership);
}

/********************************************************************
 * 函数名称: Mpi_gather
 * 功能描述:收集计算节点计算的结果
 * 参数列表:
 * 输入参数1--dDimCurrcost:计算节点得出的当前迭代所得的计算代价；
 * 输入参数2--dCurrcost:控制节点用来存储当前迭代代价；
 * 输入参数3--iNewClusterSize:计算节点得出的每个簇内数据点个数；
 * 输入参数4--iClustersize:控制节点中用于存储每个簇内数据点个数；
 * 输入参数5--dNewClusters_sum:计算节点得出的每个簇内数据点坐标和；
 * 输入参数6--dAll_newClusters_sum:控制节点中用于存储每个簇内数据点坐标和；
 * 输入参数6--dAll_newClusters_sum:控制节点中用于存储每个簇内数据点坐标和；
 * 无返回值。
 */
void Mpi_gather(double dDimCurrcost,double &dCurrcost,int *iNewClusterSize,int *&iClustersize,double **dNewClusters_sum,double **&dAll_newClusters_sum)
{
    MPI_Barrier(MPI_COMM_WORLD);//同步点
    MPI_Reduce(&dDimCurrcost,&dCurrcost,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);  //归约至进程0,进程中每个聚类的数据点与其中心点的距离之和
    MPI_Reduce(iNewClusterSize,iClustersize,iNumClusters,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);//clustersize用于存储每个簇中节点的个数

    for(int i=0;i<iNumClusters;i++)
        MPI_Reduce(dNewClusters_sum[i],dAll_newClusters_sum[i],iNumCoords,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

}

/********************************************************************
 * 函数名称: JudgeEnd
 * 功能描述:判断迭代是否终止
 * 参数列表:
 * 输入参数1--dCurrcost:当前迭代的计算代价；
 * 输入参数2--dLastcost:前一次迭代的计算代价；
 * 返回是否继续迭代。
 */
bool JudgeEnd(double dCurrcost,double dLastcost){
    if(fabs(dCurrcost-dLastcost) > dThreshold*dLastcost&& iLoop_iterations++ < MAX_Iterations)
        return true;
    else
        return false;
}

/**
* 文件:cuda.cu
* 版本: 0.0.0.1
* 作者:单晓磊
* 日期:2017/5/19
* 公司:
* 描述:k-means聚类算法所需要的聚类函数
*
* 修改记录 :
* ============================================================================
* 序号   修改日期    修改人  修改原因
*  01    2017/5/19    单晓磊  编码规范
*  02    2017/9/4    单晓磊  针对增量数据修改了SetInitMode函数以及Get_k函数
*/
#include <stdlib.h>
#include <iostream>
#include "error.h"
#include "kmeans.h"

/********************************************************************
 * 函数名称: CKmeans
 * 功能描述:构造函数
 * 参数列表:
 * 输入参数1--iNumObjs:数据点个数；
 * 输入参数2--iNumCoords:每个数据的维度；
 * 无返回值。
 */
CKmeans::CKmeans(int iNumObjs,int iNumCoords,int **iObjects){
    m_iNumObjs=iNumObjs;
	m_iNumCoords=iNumCoords;
	calltime=0;
   A=(double*)malloc(m_iNumObjs*sizeof(double));
    for(int i=0;i<m_iNumObjs;i++){
        double temp=0;
        for(int j=0;j<m_iNumCoords;j++){
            temp+=iObjects[i][j]*iObjects[i][j];
        }
        A[i]=sqrt(temp);
    }
	cudaMalloc(&deviceA,m_iNumObjs*sizeof(double));
	cudaMemcpy(deviceA,A,m_iNumObjs*sizeof(double),cudaMemcpyHostToDevice);

}

/********************************************************************
 * 函数名称: CKmeans
 * 功能描述:构造函数
 * 参数列表:
 * 输入参数1--iNumObjs:数据点个数；
 * 输入参数2--iNumCoords:每个数据的维度；
 * 输入参数3--iNumClusters:kmeans聚类个数；
 * 输入参数4--iDimobjects:用于存储数据点；
 * 无返回值。
 */
CKmeans::CKmeans(int iNumObjs,int iNumCoords,int iNumClusters,int *iDimobjects){
    m_iNumObjs=iNumObjs;
    m_iNumCoords=iNumCoords;
    m_iNumClusters=iNumClusters;

    m_iDimobjects=(int*)malloc(iNumObjs*iNumCoords*sizeof(int));
    for(int i=0;i<m_iNumObjs*m_iNumCoords;i++)
            m_iDimobjects[i]=iDimobjects[i];
    /******************************计算数据点的内积***************************************************/
    
    m_A=(double*)malloc(m_iNumObjs*sizeof(double));
    for(int i=0;i<m_iNumObjs;i++){
        double temp=0;
        for(int j=0;j<m_iNumCoords;j++){
            temp+=iDimobjects[i*m_iNumCoords+j]*iDimobjects[i*m_iNumCoords+j];
        }
        m_A[i]=sqrt(temp);
    }
    /*********************************************************************************/

    m_B=(double*)malloc(m_iNumClusters*sizeof(double));
    x=Initcuda();

}




/********************************************************************
 * 函数名称:Initcuda
 * 功能描述:对进行迭代计算时使用cuda所需要的相关参数，进行初始化和相应的空间分配

 * 返回结构体x,x包含了聚类时cuda所需的相关参数以及变量；。
 */
Cuda_p CKmeans::Initcuda()
{
 //To support reduction,iNumThreadsPerClusterBlock *must* be a power of two, and it *must* be no larger than the number of bits that will fit into an unsigned char ,the type used to keep track of iMembership changes in the kernel.
    Cuda_p x=Cuda_p(m_iNumObjs);

    cudaMalloc(&x.m_iDeviceObjects,m_iNumObjs*m_iNumCoords*sizeof(int));
    cudaMalloc(&x.m_dDeviceClusters,m_iNumClusters*m_iNumCoords*sizeof(double));
    cudaMalloc(&x.m_iDeviceMembership,m_iNumObjs*sizeof(int));
    cudaMalloc(&x.m_dDeviceIntermediates,x.m_iNumReductionThreads*sizeof(double));
    //new
    cudaMalloc(&x.m_dDeviceA,m_iNumObjs*sizeof(double));
    cudaMalloc(&x.m_dDeviceB,m_iNumClusters*sizeof(double));
    cudaMemcpy(x.m_dDeviceA,m_A,m_iNumObjs*sizeof(double),cudaMemcpyHostToDevice);
//the end
    cudaMemcpy(x.m_iDeviceObjects,m_iDimobjects,m_iNumObjs*m_iNumCoords*sizeof(int),cudaMemcpyHostToDevice);
    //CHECK_STATE("MEM");
    return x;

}
/********************************************************************
 * 函数名称: Euclid_dist_2
 * 功能描述:计算两个点的距离
 * 参数列表:
 * 输入参数1--iNumCoords:每个数据的维度；
 * 输入参数2--iNumObjs:数据点个数；
 * 输入参数3--iNumClusters:聚类个数；
 * 输入参数4--iObjects:数据点存储数组；
 * 输入参数5--dclusters:聚类中心点存储数组；
 * 输入参数6--objectId:数据点标号ID；
 * 输入参数7--clusterId:中心点标号ID；
 * 返回所计算的两个点的距离。
 */
__host__ __device__ inline static
double Euclid_dist_2(int iNumCoords,int *iObjects,double *dClusters,int objectId,int clusterId)
{
	int i;
	double ans = 0;
	for( i=0;i<iNumCoords;i++ )
	{
		ans += ( iObjects[iNumCoords * objectId + i] - dClusters[iNumCoords*clusterId + i] )*(iObjects[iNumCoords * objectId + i] - dClusters[iNumCoords*clusterId + i] ) ;
	}

        return sqrt(ans);
} 

/********************************************************************
 * 函数名称: Euclid_compare
 * 功能描述:计算两个点的相似值
 * 参数列表:
 * 输入参数1--iNumCoords:每个数据的维度；
 * 输入参数2--iNumObjs:数据点个数；
 * 输入参数3--iNumClusters:聚类个数；
 * 输入参数4--iObjects:数据点存储数组；
 * 输入参数5--dclusters:聚类中心点存储数组；
 * 输入参数6--objectId:数据点标号ID；
 * 输入参数7--clusterId:中心点标号ID；
 * 输入参数7--A:数据点内积；
 * 输入参数7--B:中心点内积；
 * 返回所计算的两个点的相似值。
 */
__host__ __device__ inline static
double Euclid_compare(int iNumCoords,int *iObjects,double *dClusters,int objectId,int clusterId,double *A,double *B)
{
	int i;
	double ans = 0;
	for( i=0;i<iNumCoords;i++ )
	{
		ans += ( iObjects[iNumCoords * objectId + i] * dClusters[iNumCoords*clusterId + i] ) ;
	}

	ans=ans/(A[objectId]*B[clusterId]);
        return ans;
}
/********************************************************************
 * 函数名称: Compute_reduction
 * 功能描述:归约已存储的数据点最短距离和
 * 参数列表:
 * 输入参数1--dDeviceIntermediates:最短距离存储数组；
 * 输入参数2--numIntermediates:数组中元素真实个数；
 * 输入参数3--numIntermediates2:大于真实个数的最小2的整数幂对应的数值；
 * 无返回值。
 */
__global__ static void Compute_reduction(double *dDeviceIntermediates,int numIntermediates,	int numIntermediates2)
{
	extern __shared__ double intermediates[];

	intermediates[threadIdx.x] = (threadIdx.x < numIntermediates) ? dDeviceIntermediates[threadIdx.x] : 0 ;
	__syncthreads();

	//numIntermediates2 *must* be a power of two!
	for(unsigned int s = numIntermediates2 /2 ; s > 0 ; s>>=1)
	{
		if(threadIdx.x < s)
		{
			intermediates[threadIdx.x] += intermediates[threadIdx.x + s];
		}
		__syncthreads();
	}

	if(threadIdx.x == 0)
	{
		dDeviceIntermediates[0] = intermediates[0];
	}
}

/********************************************************************
 * 函数名称: Find_nearest_cluster
 * 功能描述:并行计算每个数据点到各个中心点的距离，比较得出最短距离并记录每个数据点所对应中心点的标号
 * 参数列表:
 * 输入参数1--iNumCoords:每个数据的维度；
 * 输入参数2--iNumObjs:数据点个数；
 * 输入参数3--iNumClusters:聚类个数；
 * 输入参数4--iObjects:数据点存储数组；
 * 输入参数5--dDevicefClusters:聚类中心点存储数组；
 * 输入参数6--iMembership:存储每个数据点所属簇的标号；
 * 输入参数7--intermediates:存储每个block中对应的计算后的最短距离和；
 * 无返回值。
 */
__global__ static void Find_nearest_cluster(int iNumCoords,int iNumObjs,int iNumClusters,int *iObjects, double *dClusters,int *iMembership ,double *intermediates,double *A,double *B)
{
	extern __shared__ double min_distance[];
	int objectId = blockDim.x * blockIdx.x + threadIdx.x;
	double temp=0;
	while( objectId < iNumObjs )
	{
		int index;
		double dist,max_cos;
		/*find the cluster id that has min distance to iObjects*/
		index = 0;
		max_cos = Euclid_compare(iNumCoords,iObjects,dClusters,objectId,0,A,B);

		for(int i=0;i<iNumClusters;i++)
		{
			dist = Euclid_compare(iNumCoords,iObjects,dClusters,objectId,i,A,B)	;
			if( dist > max_cos )
			{
				max_cos = dist;
				index = i;
			}

		}
		temp+=max_cos;
		iMembership[objectId] = index;
		objectId+=blockDim.x * gridDim.x ;
	}
		min_distance[threadIdx.x]= temp;
		__syncthreads();

#if 1
		//blockDim.x *must* be a power of two!
		for(unsigned int s = blockDim.x / 2; s > 0 ;s>>=1)
		{
			if(threadIdx.x < s)
			{
				min_distance[threadIdx.x] += min_distance[threadIdx.x + s];//calculate all changed values and save result to iMembershipChanged[0]
			}
			__syncthreads();
		}

		if(threadIdx.x == 0)
		{
			intermediates[blockIdx.x] = min_distance[0];
		}
#endif
}//find_nearest_cluster

/********************************************************************
 * 函数名称: compute_dx
 * 功能描述:并行计算每个数据点到各个中心点的距离，比较得出最短距离并记录每个数据点所对应中心点的标号
 * 参数列表:
 * 输入参数1--iNumCoords:每个数据的维度；
 * 输入参数2--iNumObjs:数据点个数；
 * 输入参数3--iNumClusters:聚类个数；
 * 输入参数4--iObjects:数据点存储数组；
 * 输入参数5--dDevicefClusters:聚类中心点存储数组；
 * 输入参数6--d:存储每个数据点计算得出的最小距离；
 * 输入参数7--intermediates:存储每个block中对应的计算后的最短距离和；
 * 无返回值。
 */
__global__ static void  compute_dx(int iNumCoords,int iNumObjs,int iNumClusters,int *iObjects, double *dClusters,double*d,double *A,double *B)
{
	extern __shared__ double max_cos[];
	int objectId = blockDim.x * blockIdx.x + threadIdx.x;
	//double temp=0;
	if( objectId < iNumObjs )
	{
		double dist,max_cos;
		/*find the cluster id that has min distance to dObjects*/
		max_cos = Euclid_compare(iNumCoords,iObjects,dClusters,objectId,0,A,B);

		for(int i=1;i<iNumClusters;i++)
		{
			dist = Euclid_compare(iNumCoords,iObjects,dClusters,objectId,i,A,B);
			if( dist > max_cos )
				max_cos = dist;
		}

		//temp+=max_cos;
		d[objectId] = pow(max_cos,-1);
	}

}
/********************************************************************
 * 函数名称:Kmeans_cluster
 * 功能描述:依据已有中心点对所有数据点进行聚类
 * 参数列表:
 * 输入参数1--dClusters:CPU中对应存储聚类中心点的变量；
 * 输入参数2--iMembership:CPU中用于存储每个数据点所属簇的标号；
 * 输入参数3--iNewClustersize:记录每个簇对应的数据点个数；
 * 输入参数4--dNewClusters_sum:记录每个簇内所有数据点的坐标之和；
 * 返回所有数据点到当前中心点的最短距离和。
 */
 double CKmeans::Kmeans_cluster(double **dClusters,int *&iMembership,int *&iNewClustersize,double **&dNewClusters_sum)
{
    /******************************计算中心点的内积***************************************************/

    for(int i=0;i<m_iNumClusters;i++){
        double temp=0;
        for(int j=0;j<m_iNumCoords;j++){
            temp+=dClusters[i][j]*dClusters[i][j];
        }
        m_B[i]=sqrt(temp);
    }
    cudaMemcpy(x.m_dDeviceB,m_B,m_iNumClusters*sizeof(double),cudaMemcpyHostToDevice);
   
    int index;
    double dCurrcost=0;


    cudaMemcpy(x.m_iDeviceMembership,iMembership,m_iNumObjs*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(x.m_dDeviceClusters,dClusters[0],m_iNumClusters*m_iNumCoords*sizeof(double),cudaMemcpyHostToDevice);
    Find_nearest_cluster<<<x.m_iNumClusterBlocks,x.m_iNumThreadsPerClusterBlock,x.m_iClusterBlockSharedDataSize>>>(m_iNumCoords,m_iNumObjs,m_iNumClusters,x.m_iDeviceObjects,x.m_dDeviceClusters,x.m_iDeviceMembership,x.m_dDeviceIntermediates,x.m_dDeviceA,x.m_dDeviceB);
   //CHECK_STATE("Ker");
    cudaDeviceSynchronize();
    Compute_reduction<<<1,x.m_iNumReductionThreads,x.m_iReductionBlockSharedDataSize>>>(x.m_dDeviceIntermediates,x.m_iNumClusterBlocks,x.m_iNumReductionThreads);
    cudaDeviceSynchronize();
    cudaMemcpy(&dCurrcost,x.m_dDeviceIntermediates,sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(iMembership,x.m_iDeviceMembership,m_iNumObjs*sizeof(int),cudaMemcpyDeviceToHost);


    for(int i=0;i<m_iNumObjs;i++)
    {
        index = iMembership[i];

        iNewClustersize[index]++;

        for(int j=0;j<m_iNumCoords;j++)
        { 
            dNewClusters_sum[index] [j]+= m_iDimobjects[i*m_iNumCoords+j];
        }
    }

    return dCurrcost;
}

/********************************************************************
 * 函数名称: NextPowerOfTwo
 * 功能描述:计算输入整数n对应的，大于等于n的最小2的整数幂对应的数值
 * 参数列表:
 * 输入参数1--n:一个整数；
 * 返回大于等于n的最小2的整数幂对应的数值。
 */
int NextPowerOfTwo(int n)
{
	n--;
	n = n >> 1 | n;
	n = n >> 2 | n;
	n = n >> 4 | n;
	n = n >> 8 | n;
	n = n >> 16 | n;
	//n = n >> 32 | n; // for 64-bit ints
	return ++n;
}

/********************************************************************
 * 函数名称: Freecuda
 * 功能描述:释放cuda中分配的空间
 * 参数列表:
 * 输入参数1--x:结构体类型，包含聚类时cuda所需的相关参数以及变量；
 * 无返回值。
 */
void CKmeans::Freecuda()
{
	cudaFree(x.m_iDeviceObjects);
	cudaFree(x.m_dDeviceClusters);
	cudaFree(x.m_iDeviceMembership);
	cudaFree(x.m_dDeviceIntermediates);
}

/********************************************************************
 * 函数名称: Get_k
 * 功能描述:获得k-means算法中聚类个数k
 * 参数列表:
 * 输入参数1--iNumObjs:数据点个数；
 * 返回聚类个数。
 */
int CKmeans::Get_k(int iNumObjs){
    if(m_iNitMode!=InitManual){
	m_iNumClusters=sqrt(iNumObjs);
	} 
    m_iNumClusters=5;
    return m_iNumClusters;
}

/********************************************************************
 * 函数名称: SetInitMode
 * 功能描述:获得选取中心点的方式
 * 参数列表:
 * 输入参数1--i:选区中心点的方式；
 * 无返回值。
 */

void CKmeans::SetInitMode(int i){
    m_iNitMode=i;
    if(m_iNitMode==InitManual){
 	string means_file_dir = "cluster_centres.csv";
        ifstream input_file;
        input_file.open(means_file_dir.c_str(),ios::in);
        int k;
        string value;
        if(input_file.good()){
            getline(input_file,value,',');
            k=atoi(value.c_str());
        }
	input_file.close();
	m_iNumClusters=k;
}
}


/********************************************************************
 * 函数名称: SetCluster
 * 功能描述:根据所给选取中心点的方式，获得中心点
 * 参数列表:
 * 输入参数1--iObjects:CPU中用于存储数据点的变量；
 * 输入参数2--dClusters:CPU中对应存储聚类中心点的变量；
 * 有返回值,当m_iNitMode=InitManual时，返回的未k。
 */
void CKmeans::SetCluster(int **iObjects,double **&dClusters){
    // 初始化时间seed
	srand((unsigned)time(NULL));
	if(m_iNitMode ==  InitRandom) {             //随机选取中心点（不推荐）
		for(int i = 0; i <m_iNumClusters; i++) {
            int index = rand()%m_iNumObjs;
            //int index=i;
            for(int j = 0 ; j < m_iNumCoords; j++) {
                dClusters[i][j] = iObjects[index][j]; // 这里不知道真实数据中词频的值域，直接随机容易远离真实数据点
            }
		}
	}
	else if(m_iNitMode == InitScreen) {        //  从N个输入数据选k个设为初始中心点
		int k = m_iNumClusters;
		for(int i=0;i<m_iNumObjs;i++){
		    if(i<k) {
                for(int j = 0 ; j < m_iNumCoords; j++)
                    dClusters[i][j] = iObjects[i][j];
                continue;
		    }
            int n = i+1,id = rand()%1000;
            double eps = 1e-6;
            if(double(id+1)/1000 < k*1.0/n + eps){
                int swap_id = rand() % k;
                for(int j = 0 ; j < m_iNumCoords; j++) {
                    dClusters[swap_id][j] =iObjects[i][j];
                    }
            }
		}
	}
	else if(m_iNitMode == InitManual) {//手动选取 从文件中选取（当使用了增量数据的时候建议使用。这里仅提供一种简单实现）
        string means_file_dir = "cluster_centres.csv";
        ifstream input_file;
        input_file.open(means_file_dir.c_str(),ios::in);
        int k,dim;
        string value;
        if(input_file.good()){
            getline(input_file,value,',');
            k=atoi(value.c_str());
            getline(input_file,value,',');
            dim=atoi(value.c_str());
        }
            for(int i=0;i<k;i++) {
                double x;
                for(int j=0;j<dim;j++) {
                    getline(input_file,value,',');
                    x=atof(value.c_str());
                    dClusters[i][j] = x;
                }
            }
        input_file.close();
	}

	else if(m_iNitMode == Initkmeans2){//利用kmeans++算法选取输入的k个文档作为初始中心集
        double *d;
     	d=(double*)malloc(m_iNumObjs*sizeof(double));
     	memset(d,0,m_iNumObjs*sizeof(double));
   
	const unsigned int iNumThreadsPerBlock=(m_iNumObjs>128?1024:128);
	const unsigned int iNumBlocks = imin(512,(m_iNumObjs + iNumThreadsPerBlock -1)/iNumThreadsPerBlock);
	const unsigned int iNumReduction=NextPowerOfTwo(iNumBlocks);
	const unsigned int BlockSharedDataSize=iNumThreadsPerBlock*sizeof(double);
	const unsigned int reductionSharedDataSize=iNumReduction*sizeof(double);
	double *deviceIntermediates;
	double *devicedimClusters;
	double *deviced;
	int *deviceObjects;
	cudaMalloc(&deviceObjects,m_iNumObjs*m_iNumCoords*sizeof(int));
    	cudaMemcpy(deviceObjects,iObjects[0],m_iNumObjs*m_iNumCoords*sizeof(int),cudaMemcpyHostToDevice);
	cudaMalloc(&deviceIntermediates,iNumReduction*sizeof(double));
	cudaMalloc(&devicedimClusters,m_iNumClusters*m_iNumCoords*sizeof(double));
	cudaMalloc(&deviced,m_iNumObjs*sizeof(double));
     	int k;
     	double sum;
	srand((int)time(NULL));
     	int basenum=rand() % m_iNumObjs;
	
    	//随机选取初始第一个中心点

    	 for(int i=0;i<m_iNumCoords;i++){
         	dClusters[0][i]=iObjects[basenum][i];
    	 }
	/*****************************************/
	 double *B;
         B=(double*)malloc(m_iNumClusters*sizeof(double));
         memset(B,0,m_iNumClusters*sizeof(double));
         double temp=0;
         for(int i=0;i<m_iNumCoords;i++)
            temp+=dClusters[0][i]*dClusters[0][i];
         B[0]=sqrt(temp);
double *deviceB;

	cudaMalloc(&deviceB,m_iNumClusters*sizeof(double));
	cudaMemcpy(deviceB,B,m_iNumClusters*sizeof(double),cudaMemcpyHostToDevice);

	/*****************************************/
     	for(k=1;k<m_iNumClusters;k++){
         sum=0;
         //计算所有数据点到中心点的最短距离
		
	cudaMemcpy(devicedimClusters,dClusters[0],m_iNumClusters*m_iNumCoords*sizeof(double),cudaMemcpyHostToDevice);
	compute_dx<<<iNumBlocks,iNumThreadsPerBlock,BlockSharedDataSize>>>(m_iNumCoords,m_iNumObjs,k,deviceObjects,devicedimClusters,deviced,deviceA,deviceB);
	//CHECK_STATE("Ker");
	cudaDeviceSynchronize();
	Compute_reduction<<<1,iNumReduction,reductionSharedDataSize>>>(deviceIntermediates,iNumBlocks,iNumReduction);
	cudaDeviceSynchronize();
	cudaMemcpy(&sum,deviceIntermediates,sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(d,deviced,m_iNumObjs*sizeof(double),cudaMemcpyDeviceToHost);
	
	
       /*  for(int j=0;j<m_iNumObjs;j++)
             p[j]=d[j]/sum;
*/
         sum=sum * rand() / (RAND_MAX - 1.);
         int max=0;
	// double temp=d[0];
         for(int j=1;j<m_iNumObjs;j++){
		if((sum -= d[j]) > 0)
			 continue;
		max=j;
		break;
		
	  }
	
 	for(int i=0;i<m_iNumCoords;i++)
                 dClusters[k][i]=iObjects[max][i];
	temp=0;
        for(int i=0;i<m_iNumCoords;i++)
            temp+=dClusters[k][i]*dClusters[k][i];
        B[k]=sqrt(temp);

	cudaMemcpy(deviceB,B,m_iNumClusters*sizeof(double),cudaMemcpyHostToDevice);
}
     
	cudaFree(deviceIntermediates);
	cudaFree(deviceObjects);
	cudaFree(devicedimClusters);
	cudaFree(deviced);
	cudaFree(deviceB);
     	free(d);
	free(B);

}
//测试
    /*   for(int i=0;i<m_iNumClusters;i++)
            for(int j=0;j<m_iNumCoords;j++)
                dClusters[i][j]=iObjects[i][j];*/
}


/********************************************************************
 * 函数名称: Updatecenter
 * 功能描述:更新中心点坐标
 * 参数列表:
 * 输入参数1--iNumClusters:聚类个数；
 * 输入参数2--iNumObjs:数据点个数；
 * 输入参数3--iNumCoords:每个数据的维度；
 * 输入参数4--dClusters:CPU中对应存储聚类中心点的变量；
 * 输入参数5--iClustersize:记录每个簇对应的数据点个数；
 * 输入参数6--dAll_newClusters_sum:记录每个簇内所有数据点的坐标之和
 * 无返回值。
 */
void CKmeans::Updatecenter(double **&dClusters,int *iClustersize,double **dAll_newClusters_sum)
{    		//average the sum and replace old cluster centers with newfClusters
    for(int i=0;i<m_iNumClusters;i++)
        for(int j=0;j<m_iNumCoords;j++)
        {
            if(iClustersize[i] > 0)
                dClusters[i][j] = dAll_newClusters_sum[i][j]/iClustersize[i];
        }
}


double CKmeans::together(double **dClusters,int *iMembership){
	double sum=0;
	int index=-1;
	for(int i=0;i<m_iNumObjs;i++){		
		double temp=0;
		index=iMembership[i];
		for(int j=0;j<m_iNumCoords;j++)
			temp+=(m_iDimobjects[i*m_iNumCoords+j]*dClusters[index][j]);
		temp=temp/(m_A[i]*m_B[index]);
		sum+=temp;
	}
return sum;
}

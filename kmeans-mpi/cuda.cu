/**
* �ļ�:cuda.cu
* �汾: 0.0.0.1
* ����:������
* ����:2017/5/19
* ��˾:
* ����:k-means�����㷨����Ҫ�ľ��ຯ��
*
* �޸ļ�¼ :
* ============================================================================
* ���   �޸�����    �޸���  �޸�ԭ��
*  01    2017/5/19    ������  ����淶
*  02    2017/9/4    ������  ������������޸���SetInitMode�����Լ�Get_k����
*/
#include <stdlib.h>
#include <iostream>
#include "error.h"
#include "kmeans.h"

/********************************************************************
 * ��������: CKmeans
 * ��������:���캯��
 * �����б�:
 * �������1--iNumObjs:���ݵ������
 * �������2--iNumCoords:ÿ�����ݵ�ά�ȣ�
 * �޷���ֵ��
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
 * ��������: CKmeans
 * ��������:���캯��
 * �����б�:
 * �������1--iNumObjs:���ݵ������
 * �������2--iNumCoords:ÿ�����ݵ�ά�ȣ�
 * �������3--iNumClusters:kmeans���������
 * �������4--iDimobjects:���ڴ洢���ݵ㣻
 * �޷���ֵ��
 */
CKmeans::CKmeans(int iNumObjs,int iNumCoords,int iNumClusters,int *iDimobjects){
    m_iNumObjs=iNumObjs;
    m_iNumCoords=iNumCoords;
    m_iNumClusters=iNumClusters;

    m_iDimobjects=(int*)malloc(iNumObjs*iNumCoords*sizeof(int));
    for(int i=0;i<m_iNumObjs*m_iNumCoords;i++)
            m_iDimobjects[i]=iDimobjects[i];
    /******************************�������ݵ���ڻ�***************************************************/
    
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
 * ��������:Initcuda
 * ��������:�Խ��е�������ʱʹ��cuda����Ҫ����ز��������г�ʼ������Ӧ�Ŀռ����

 * ���ؽṹ��x,x�����˾���ʱcuda�������ز����Լ���������
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
 * ��������: Euclid_dist_2
 * ��������:����������ľ���
 * �����б�:
 * �������1--iNumCoords:ÿ�����ݵ�ά�ȣ�
 * �������2--iNumObjs:���ݵ������
 * �������3--iNumClusters:���������
 * �������4--iObjects:���ݵ�洢���飻
 * �������5--dclusters:�������ĵ�洢���飻
 * �������6--objectId:���ݵ���ID��
 * �������7--clusterId:���ĵ���ID��
 * �����������������ľ��롣
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
 * ��������: Euclid_compare
 * ��������:���������������ֵ
 * �����б�:
 * �������1--iNumCoords:ÿ�����ݵ�ά�ȣ�
 * �������2--iNumObjs:���ݵ������
 * �������3--iNumClusters:���������
 * �������4--iObjects:���ݵ�洢���飻
 * �������5--dclusters:�������ĵ�洢���飻
 * �������6--objectId:���ݵ���ID��
 * �������7--clusterId:���ĵ���ID��
 * �������7--A:���ݵ��ڻ���
 * �������7--B:���ĵ��ڻ���
 * ����������������������ֵ��
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
 * ��������: Compute_reduction
 * ��������:��Լ�Ѵ洢�����ݵ���̾����
 * �����б�:
 * �������1--dDeviceIntermediates:��̾���洢���飻
 * �������2--numIntermediates:������Ԫ����ʵ������
 * �������3--numIntermediates2:������ʵ��������С2�������ݶ�Ӧ����ֵ��
 * �޷���ֵ��
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
 * ��������: Find_nearest_cluster
 * ��������:���м���ÿ�����ݵ㵽�������ĵ�ľ��룬�Ƚϵó���̾��벢��¼ÿ�����ݵ�����Ӧ���ĵ�ı��
 * �����б�:
 * �������1--iNumCoords:ÿ�����ݵ�ά�ȣ�
 * �������2--iNumObjs:���ݵ������
 * �������3--iNumClusters:���������
 * �������4--iObjects:���ݵ�洢���飻
 * �������5--dDevicefClusters:�������ĵ�洢���飻
 * �������6--iMembership:�洢ÿ�����ݵ������صı�ţ�
 * �������7--intermediates:�洢ÿ��block�ж�Ӧ�ļ�������̾���ͣ�
 * �޷���ֵ��
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
 * ��������: compute_dx
 * ��������:���м���ÿ�����ݵ㵽�������ĵ�ľ��룬�Ƚϵó���̾��벢��¼ÿ�����ݵ�����Ӧ���ĵ�ı��
 * �����б�:
 * �������1--iNumCoords:ÿ�����ݵ�ά�ȣ�
 * �������2--iNumObjs:���ݵ������
 * �������3--iNumClusters:���������
 * �������4--iObjects:���ݵ�洢���飻
 * �������5--dDevicefClusters:�������ĵ�洢���飻
 * �������6--d:�洢ÿ�����ݵ����ó�����С���룻
 * �������7--intermediates:�洢ÿ��block�ж�Ӧ�ļ�������̾���ͣ�
 * �޷���ֵ��
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
 * ��������:Kmeans_cluster
 * ��������:�����������ĵ���������ݵ���о���
 * �����б�:
 * �������1--dClusters:CPU�ж�Ӧ�洢�������ĵ�ı�����
 * �������2--iMembership:CPU�����ڴ洢ÿ�����ݵ������صı�ţ�
 * �������3--iNewClustersize:��¼ÿ���ض�Ӧ�����ݵ������
 * �������4--dNewClusters_sum:��¼ÿ�������������ݵ������֮�ͣ�
 * �����������ݵ㵽��ǰ���ĵ����̾���͡�
 */
 double CKmeans::Kmeans_cluster(double **dClusters,int *&iMembership,int *&iNewClustersize,double **&dNewClusters_sum)
{
    /******************************�������ĵ���ڻ�***************************************************/

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
 * ��������: NextPowerOfTwo
 * ��������:������������n��Ӧ�ģ����ڵ���n����С2�������ݶ�Ӧ����ֵ
 * �����б�:
 * �������1--n:һ��������
 * ���ش��ڵ���n����С2�������ݶ�Ӧ����ֵ��
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
 * ��������: Freecuda
 * ��������:�ͷ�cuda�з���Ŀռ�
 * �����б�:
 * �������1--x:�ṹ�����ͣ���������ʱcuda�������ز����Լ�������
 * �޷���ֵ��
 */
void CKmeans::Freecuda()
{
	cudaFree(x.m_iDeviceObjects);
	cudaFree(x.m_dDeviceClusters);
	cudaFree(x.m_iDeviceMembership);
	cudaFree(x.m_dDeviceIntermediates);
}

/********************************************************************
 * ��������: Get_k
 * ��������:���k-means�㷨�о������k
 * �����б�:
 * �������1--iNumObjs:���ݵ������
 * ���ؾ��������
 */
int CKmeans::Get_k(int iNumObjs){
    if(m_iNitMode!=InitManual){
	m_iNumClusters=sqrt(iNumObjs);
	} 
    m_iNumClusters=5;
    return m_iNumClusters;
}

/********************************************************************
 * ��������: SetInitMode
 * ��������:���ѡȡ���ĵ�ķ�ʽ
 * �����б�:
 * �������1--i:ѡ�����ĵ�ķ�ʽ��
 * �޷���ֵ��
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
 * ��������: SetCluster
 * ��������:��������ѡȡ���ĵ�ķ�ʽ��������ĵ�
 * �����б�:
 * �������1--iObjects:CPU�����ڴ洢���ݵ�ı�����
 * �������2--dClusters:CPU�ж�Ӧ�洢�������ĵ�ı�����
 * �з���ֵ,��m_iNitMode=InitManualʱ�����ص�δk��
 */
void CKmeans::SetCluster(int **iObjects,double **&dClusters){
    // ��ʼ��ʱ��seed
	srand((unsigned)time(NULL));
	if(m_iNitMode ==  InitRandom) {             //���ѡȡ���ĵ㣨���Ƽ���
		for(int i = 0; i <m_iNumClusters; i++) {
            int index = rand()%m_iNumObjs;
            //int index=i;
            for(int j = 0 ; j < m_iNumCoords; j++) {
                dClusters[i][j] = iObjects[index][j]; // ���ﲻ֪����ʵ�����д�Ƶ��ֵ��ֱ���������Զ����ʵ���ݵ�
            }
		}
	}
	else if(m_iNitMode == InitScreen) {        //  ��N����������ѡk����Ϊ��ʼ���ĵ�
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
	else if(m_iNitMode == InitManual) {//�ֶ�ѡȡ ���ļ���ѡȡ����ʹ�����������ݵ�ʱ����ʹ�á�������ṩһ�ּ�ʵ�֣�
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

	else if(m_iNitMode == Initkmeans2){//����kmeans++�㷨ѡȡ�����k���ĵ���Ϊ��ʼ���ļ�
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
	
    	//���ѡȡ��ʼ��һ�����ĵ�

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
         //�����������ݵ㵽���ĵ����̾���
		
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
//����
    /*   for(int i=0;i<m_iNumClusters;i++)
            for(int j=0;j<m_iNumCoords;j++)
                dClusters[i][j]=iObjects[i][j];*/
}


/********************************************************************
 * ��������: Updatecenter
 * ��������:�������ĵ�����
 * �����б�:
 * �������1--iNumClusters:���������
 * �������2--iNumObjs:���ݵ������
 * �������3--iNumCoords:ÿ�����ݵ�ά�ȣ�
 * �������4--dClusters:CPU�ж�Ӧ�洢�������ĵ�ı�����
 * �������5--iClustersize:��¼ÿ���ض�Ӧ�����ݵ������
 * �������6--dAll_newClusters_sum:��¼ÿ�������������ݵ������֮��
 * �޷���ֵ��
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

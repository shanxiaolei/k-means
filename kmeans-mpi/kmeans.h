/**
* 文件:kmeans.h
* 版本: 0.0.0.1
* 作者:单晓磊
* 日期:2017/5/19
* 公司:
* 描述:k-means聚类算法所需要的函数声明
*
* 修改记录 :
* ============================================================================
* 序号   修改日期    修改人  修改原因
*  01    2017/5/19    单晓磊  编码规范
*/

#ifndef _H_KMEANS
#define _H_KMEANS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>
#include <cassert>
#include <fstream>
#include <math.h>
#include <time.h>


#define MAX_Iterations 500
#define MAX_CHAR_PER_LINE 1024000
#define malloc2D(name, xDim, yDim, type) do {               \
    name = (type **)malloc(xDim * sizeof(type *));          \
    assert(name != NULL);                                   \
    name[0] = (type *)malloc(xDim * yDim * sizeof(type));   \
    assert(name[0] != NULL);                                \
    for (size_t i = 1; i < xDim; i++)                       \
        name[i] = name[i-1] + yDim;                         \
} while (0)


#define imin(a,b) (a<b?a:b)

using namespace std;

int NextPowerOfTwo(int iNum);

struct Cuda_p{
    int  *m_iDeviceObjects; //[iNumCoords][iNumObjs]
    double *m_dDeviceClusters; //[iNumCoords][numclusters]
    int *m_iDeviceMembership;
    double *m_dDeviceIntermediates;
    int m_iNumThreadsPerClusterBlock ;
    int m_iNumClusterBlocks ;
    int m_iNumReductionThreads;
    int m_iClusterBlockSharedDataSize;
    int m_iReductionBlockSharedDataSize;
//new
    double *m_dDeviceA;
    double *m_dDeviceB;
    Cuda_p() {}
    ~Cuda_p() {}
    Cuda_p(int iNumObjs){
        m_iNumThreadsPerClusterBlock = (iNumObjs>128?1024:128);
        m_iNumClusterBlocks = imin(32,(iNumObjs + m_iNumThreadsPerClusterBlock -1)/m_iNumThreadsPerClusterBlock);
        m_iNumReductionThreads = NextPowerOfTwo(m_iNumClusterBlocks);
        m_iClusterBlockSharedDataSize = m_iNumThreadsPerClusterBlock * sizeof(double);
        m_iReductionBlockSharedDataSize = m_iNumReductionThreads * sizeof(double);
    }

};

class CKmeans{
public:
	enum InitMode {     //  中心集初始化模式
		InitRandom,     //  随机模式
		InitManual,     //  手动模式
		InitScreen,    //  选取输入数据中的k个文档当做中心集
		Initkmeans2,//利用kmeans++算法选取输入的k个文档作为初始中心集
	};
    CKmeans() {}
	CKmeans(int iNumObjs,int iNumCoords,int **iObjects);
	CKmeans(int iNumObjs,int iNumCoords,int iNumClusters,int *dDimobjects);
	~CKmeans(){}
    void SetCluster(int **dObjects,double **&dClusters);
    Cuda_p Initcuda();
    int Get_k(int iNumObjs);
    void SetInitMode(int i);
    double Kmeans_cluster(double **dClusters,int *&iMembership,int *&iNewClustersize,double**&dNewClusters_sum);
    void Freecuda();
    void Updatecenter(double **&dClusters,int *iClustersize,double **dAll_newClusters_sum);
double together(double **dClusters,int *iMembership);
    void Free();
private:
    int m_iNumClusters; //聚类中心点的个数
    int m_iNumObjs;      //数据点的个数
	int m_iNumCoords;      //数据点的维度
	int *m_iDimobjects;    //存取中心点的数组
   double calltime;
    Cuda_p x;               //cuda执行的相关参数
	int m_iNitMode;         // 中心集初始化方式    默认: InitRandom
double *m_A;
double *m_B;
double *A;
double *deviceA;

};


//错误处理
#define CHECK_ERROR(error) checkCudaError(error, __FILE__, __LINE__)
#define CHECK_STATE(msg) checkCudaState(msg, __FILE__, __LINE__)
/*
inline void checkCudaError(cudaError_t error, const char *file, const int line)
{
   if (error != cudaSuccess) {
      std::cerr << "CUDA CALL FAILED:" << file << "( " << line << " )- " << cudaGetErrorString(error) << std::endl;
      exit(EXIT_FAILURE);

   }

}
inline void checkCudaState(const char *msg, const char *file, const int line)
{
   cudaError_t error = cudaGetLastError();
   if (error != cudaSuccess) {
      std::cerr << "---" << msg << " Error---" << std::endl;
      std::cerr << file << "( " << line << " )- " << cudaGetErrorString(error) << std::endl;
      exit(EXIT_FAILURE);

   }
}
*/

int File_read(int& iNumObjs,int& iNumCoords,int **&iObjects,char *cFilename);
int File_write(int iNumObjs,int iNumCoords,int iNumClusters,int *iMembership,double **dClusters);
void Resetvar(int iNumClusters,int iNumCoords,int *&iNewClusterSize,double **&dNewClusters_sum,int *&iClustersize);//

#endif


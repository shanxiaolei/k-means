/**
* �ļ�:kmeans.h
* �汾: 0.0.0.1
* ����:������
* ����:2017/5/19
* ��˾:
* ����:k-means�����㷨����Ҫ�ĺ�������
*
* �޸ļ�¼ :
* ============================================================================
* ���   �޸�����    �޸���  �޸�ԭ��
*  01    2017/5/19    ������  ����淶
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
	enum InitMode {     //  ���ļ���ʼ��ģʽ
		InitRandom,     //  ���ģʽ
		InitManual,     //  �ֶ�ģʽ
		InitScreen,    //  ѡȡ���������е�k���ĵ��������ļ�
		Initkmeans2,//����kmeans++�㷨ѡȡ�����k���ĵ���Ϊ��ʼ���ļ�
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
    int m_iNumClusters; //�������ĵ�ĸ���
    int m_iNumObjs;      //���ݵ�ĸ���
	int m_iNumCoords;      //���ݵ��ά��
	int *m_iDimobjects;    //��ȡ���ĵ������
   double calltime;
    Cuda_p x;               //cudaִ�е���ز���
	int m_iNitMode;         // ���ļ���ʼ����ʽ    Ĭ��: InitRandom
double *m_A;
double *m_B;
double *A;
double *deviceA;

};


//������
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


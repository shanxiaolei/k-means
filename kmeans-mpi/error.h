/**
* �ļ�:kmeans.h
* �汾: 0.0.0.1
* ����:������
* ����:2017/5/19
* ��˾:
* ����:k-means�����㷨������궨��
*
* �޸ļ�¼ :
* ============================================================================
* ���   �޸�����    �޸���  �޸�ԭ��
*  01    2017/5/19    ������  ����淶
*  02    2017/6/10    ������  ����淶�����������궨��
*/
#include <assert.h>
#include<iostream>
using namespace std;

#define E_CUDAMALLOC 200001  //CUDA�з���ռ����
#define E_CUDACOPY   200002   //CPU��GPU�������ݳ���
#define E_CUDDPARAMETER 200003//����CUDAʱ�������ã��߳̿���Ŀ���߳���Ŀ������
#define E_CALLCUDAPAR 200004 //����CUDAʱ��������
#define E_FILEREAD 200005 //��ȡ���ݳ���
#define E_FILWRITE 200006 //д�����ݳ���

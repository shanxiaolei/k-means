/**
* �ļ�:mpi_io.cu
* �汾: 0.0.0.1
* ����:������
* ����:2017/5/19
* ��˾:
* ����:k-means�����㷨��Ӧ���ļ���дIO
*
* �޸ļ�¼ :
* ============================================================================
* ���   �޸�����    �޸���  �޸�ԭ��
*  01    2017/5/19    ������  ����淶
*  02    2017/6/10    ������  ����淶
*/
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "kmeans.h"
#include "error.h"
using namespace std;
/********************************************************************
 * ��������: File_read
 * ��������:�����ļ��������
 * �����б�:
 * �������1--iNumObjs:���ݵ������
 * �������2--iNumCoords:ÿ�����ݵ�ά�ȣ�
 * �������3--iObjects:���ݵ�洢���飻
 * �������4--cFilename���ļ����֣�
 * �޷���ֵ��
 */
int File_read(int& iNumObjs,int& iNumCoords,int **&iObjects,char *cFilename)
{
	FILE *infile;
	char *line = new char[MAX_CHAR_PER_LINE];
	int lineLen = MAX_CHAR_PER_LINE;

	infile = fopen(cFilename,"r");
	assert(infile!=NULL);
	/*find the number of objects*/
	while( fgets(line,lineLen,infile) )
	{
		iNumObjs++;
	}

	/*find the dimension of each object*/
	rewind(infile);
//numCoords=1;
	while( fgets(line,lineLen,infile)!=NULL )
	{
		if( strtok(line," ")!=0 )
		{
			while( strtok(NULL," ") )
				iNumCoords++;
			break;
		}
	}

	/*allocate space for object[][] and read all objcet*/
	rewind(infile);

	malloc2D(iObjects,iNumObjs,iNumCoords,int);
	int i=0;
	/*read all object*/
	while( fgets(line,lineLen,infile)!=NULL )
	{
		iObjects[i][0]=atoi(strtok(line," ") );
		for(int j=1;j<iNumCoords;j++)
		{
			iObjects[i][j] = atoi( strtok(NULL," ") )	;
		}
		i++;
	}
	return 0;

/*
    	cudaMalloc(&deviceObjects,numObjs*numCoords*sizeof(int));
    	cudaMemcpy(deviceObjects,objects[0],numObjs*numCoords*sizeof(int),cudaMemcpyHostToDevice);
	cudaMalloc(&deviceMembership,numObjs*sizeof(int));
	cudaMemcpy(deviceMembership,membership,numObjs*sizeof(int),cudaMemcpyHostToDevice);
*/
}

/********************************************************************
 * ��������: File_write
 * ��������:��������д���ļ�
 * �����б�:
 * �������1--iNumObjs:���ݵ������
 * �������2--iNumCoords:ÿ�����ݵ�ά�ȣ�
 * �������3--iMembership:���ݵ�洢���飻
 * �������4--dClusters���ļ����֣�
 * �޷���ֵ��
 */
int File_write(int iNumObjs,int iNumCoords,int iNumClusters,int *iMembership,double **dClusters)
{
    //��������д�롰kmeans.csv���ļ���
    char file_name[]="kmeans.csv";
    ofstream output;
    output.open(file_name,ios::out);
    if(output.fail())
    {
 		cout << "�ļ���ʧ��" << endl;
        //MDNOV_Error("Error:open file wrong %d",E_FILWRITE);
		return 0;
    }
    for(int i = 0 ; i < iNumClusters ; i++) {
        for(int j = 0 ; j <iNumObjs; j++) {
            if(iMembership[j]==i)
                output <<j<<",";
        }
        output << endl;
    }
    output.close();
    //���������ģ�д�뵽cluster_centres.csv��ȥ
    char file_name1[]="cluster_centres.csv";
    ofstream output1;
    output1.open(file_name1,ios::out);
    output1<<iNumClusters<<","<<iNumCoords<<endl;
    for(int i=0;i<iNumClusters;i++){
	for(int j=0;j<iNumCoords;j++)
	    output1<<dClusters[i][j]<<",";
	output1<<endl;
    }
    output1.close();


    //д�뵽iMembership.csv��ȥ
    char file_name2[]="iMembership.csv";
    ofstream output2;
    output2.open(file_name2,ios::out);
    for(int i=0;i<iNumObjs;i++)
	    output2<<iMembership[i]<<",";
    output2.close();

    return 0;
}

/********************************************************************
 * ��������: Resetvar
 * ��������:��������
 * �����б�:
 * �������1--iNumClusters:�������ĵ������
 * �������2--iNumCoords:ÿ�����ݵ�ά�ȣ�
 * �������3--iNewClustersize:��¼�ֲ�ÿ���ض�Ӧ�����ݵ������
 * �������4--iNewClusters_sum:��¼�ֲ�ÿ�������������ݵ������֮�ͣ�
 * �������3--iClustersize:��¼����ÿ���ض�Ӧ�����ݵ������
 * �޷���ֵ��
 */
void Resetvar(int iNumClusters,int iNumCoords,int *&iNewClusterSize,double **&dNewClusters_sum,int *&iClustersize){
    //��newClusters������ͣ��Լ�iNewClusterSize��ÿһ�������ݵ����Ŀ����Ϊ0
    for(int i=0;i<iNumClusters;i++)
    {
        for(int j=0;j<iNumCoords;j++)
            dNewClusters_sum[i][j]=0.0;
        iNewClusterSize[i]=0;
        iClustersize[i]=0;
    }
}

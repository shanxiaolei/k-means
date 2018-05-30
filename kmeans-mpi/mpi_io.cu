/**
* 文件:mpi_io.cu
* 版本: 0.0.0.1
* 作者:单晓磊
* 日期:2017/5/19
* 公司:
* 描述:k-means聚类算法对应的文件读写IO
*
* 修改记录 :
* ============================================================================
* 序号   修改日期    修改人  修改原因
*  01    2017/5/19    单晓磊  编码规范
*  02    2017/6/10    单晓磊  编码规范
*/
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "kmeans.h"
#include "error.h"
using namespace std;
/********************************************************************
 * 函数名称: File_read
 * 功能描述:读入文件获得数据
 * 参数列表:
 * 输入参数1--iNumObjs:数据点个数；
 * 输入参数2--iNumCoords:每个数据的维度；
 * 输入参数3--iObjects:数据点存储数组；
 * 输入参数4--cFilename：文件名字；
 * 无返回值。
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
 * 函数名称: File_write
 * 功能描述:将聚类结果写入文件
 * 参数列表:
 * 输入参数1--iNumObjs:数据点个数；
 * 输入参数2--iNumCoords:每个数据的维度；
 * 输入参数3--iMembership:数据点存储数组；
 * 输入参数4--dClusters：文件名字；
 * 无返回值。
 */
int File_write(int iNumObjs,int iNumCoords,int iNumClusters,int *iMembership,double **dClusters)
{
    //将聚类结果写入“kmeans.csv”文件中
    char file_name[]="kmeans.csv";
    ofstream output;
    output.open(file_name,ios::out);
    if(output.fail())
    {
 		cout << "文件打开失败" << endl;
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
    //将聚类中心，写入到cluster_centres.csv中去
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


    //写入到iMembership.csv中去
    char file_name2[]="iMembership.csv";
    ofstream output2;
    output2.open(file_name2,ios::out);
    for(int i=0;i<iNumObjs;i++)
	    output2<<iMembership[i]<<",";
    output2.close();

    return 0;
}

/********************************************************************
 * 函数名称: Resetvar
 * 功能描述:变量重置
 * 参数列表:
 * 输入参数1--iNumClusters:聚类中心点个数；
 * 输入参数2--iNumCoords:每个数据的维度；
 * 输入参数3--iNewClustersize:记录局部每个簇对应的数据点个数；
 * 输入参数4--iNewClusters_sum:记录局部每个簇内所有数据点的坐标之和；
 * 输入参数3--iClustersize:记录整体每个簇对应的数据点个数；
 * 无返回值。
 */
void Resetvar(int iNumClusters,int iNumCoords,int *&iNewClusterSize,double **&dNewClusters_sum,int *&iClustersize){
    //将newClusters（坐标和）以及iNewClusterSize（每一类中数据点的数目）设为0
    for(int i=0;i<iNumClusters;i++)
    {
        for(int j=0;j<iNumCoords;j++)
            dNewClusters_sum[i][j]=0.0;
        iNewClusterSize[i]=0;
        iClustersize[i]=0;
    }
}

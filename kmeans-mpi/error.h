/**
* 文件:kmeans.h
* 版本: 0.0.0.1
* 作者:单晓磊
* 日期:2017/5/19
* 公司:
* 描述:k-means聚类算法错误处理宏定义
*
* 修改记录 :
* ============================================================================
* 序号   修改日期    修改人  修改原因
*  01    2017/5/19    单晓磊  编码规范
*  02    2017/6/10    单晓磊  编码规范，加入错误处理宏定义
*/
#include <assert.h>
#include<iostream>
using namespace std;

#define E_CUDAMALLOC 200001  //CUDA中分配空间出错
#define E_CUDACOPY   200002   //CPU与GPU拷贝数据出错
#define E_CUDDPARAMETER 200003//调用CUDA时参数配置（线程块数目、线程数目）出错
#define E_CALLCUDAPAR 200004 //调用CUDA时参数出错
#define E_FILEREAD 200005 //读取数据出错
#define E_FILWRITE 200006 //写入数据出错

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <stdio.h>
//#include <tchar.h>
//#include <fstream>
//#define CV_VERSION_ID       CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
//#ifdef _DEBUG
//#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
//#else
//#define cvLIB(name) "opencv_" name CV_VERSION_ID
//#endif
//#pragma comment( lib, cvLIB("core") )
//#pragma comment( lib, cvLIB("imgproc") )
//#pragma comment( lib, cvLIB("highgui") )
//#pragma comment( lib, cvLIB("flann") )
//#pragma comment( lib, cvLIB("features2d") )
//#pragma comment( lib, cvLIB("calib3d") )
//#pragma comment( lib, cvLIB("gpu") )
//#pragma comment( lib, cvLIB("legacy") )
//#pragma comment( lib, cvLIB("ml") )
//#pragma comment( lib, cvLIB("objdetect") )
//#pragma comment( lib, cvLIB("ts") )
//#pragma comment( lib, cvLIB("video") )
//#pragma comment( lib, cvLIB("contrib") )
//#pragma comment( lib, cvLIB("nonfree") )
 using namespace cv;
 using namespace std;



 //atan2函数
Mat cvAtan2Mat(Mat a, Mat b)
{
    int rows = a.rows;
    int cols = a.cols;
    Mat out;
	out.create(rows, cols, CV_64FC1);
   // for (int i=0;i<rows;i++)  
   // {
   //     for (int j=0;j<cols;j++)  
   //     {
			//double ptra=a.at<double>(i,j);
			//double ptrb=b.at<double>(i,j);
			//double ptrout=atan2(ptra,ptrb);
			////cout<<"ptrout="<<ptrout<<endl;
			//out.at<double>(i,j)=ptrout;    //修改灰度值  
   //     }  
   // }    
	{  
		// 取源图像的指针  
		const double* ptra = a.ptr<double>(0); 
		const double* ptrb = b.ptr<double>(0);
		// 将输出数据指针存放输出图像  
		double* ptrout = out.ptr<double>(0);  
		for(int j=0; j < cols*rows; j++)  
		{  
			ptrout[j]= cv::saturate_cast<double>(atan2(ptra[j],ptrb[j]));  
		}  
	} 
	return out;
    
}

//c=0,sin函数;c=1,cos函数
Mat cvSinMat(Mat a,int c)
{
    int rows = a.rows;
    int cols = a.cols;
    Mat out;
	out.create(rows, cols, CV_64FC1);      
	{  
		// 取源图像的指针  
		const double* ptra = a.ptr<double>(0); 
		// 将输出数据指针存放输出图像  
		double* ptrout = out.ptr<double>(0);  
		for(int j=0; j < cols*rows; j++)  
		{
			if(c==0)
			ptrout[j]= cv::saturate_cast<double>(sin(ptra[j]));
			else ptrout[j]= cv::saturate_cast<double>(cos(ptra[j]));
		}  
	} 
	return out;
    
}

//angle函数
Mat cvAngleMat(Mat a)
  {
	Mat realf=cvSinMat(a,1);
	Mat imagf=cvSinMat(a,0);
	Mat out=cvAtan2Mat(imagf,realf);
	return out;
  }

 //滤波函数
Mat cvFilMat(Mat a)
  {

	Mat realf=cvSinMat(a,1);
	Mat imagf=cvSinMat(a,0);
	for (int i=0;i<10;i++)
	{
	//medianBlur(cvCosMat(wrapped_phase),realf,3);//中值滤波
	//medianBlur(cvSinMat(wrapped_phase),imagf,3);
	blur(realf,realf,Size(7,7));//均值滤波
	blur(imagf,imagf,Size(7,7));
	}
	Mat out=cvAtan2Mat(imagf,realf);
	return out;
  }

//dx差分,c=0：第1行减第0行，放在第0行，依此类推，最后一行全为0；
//c=1：第1行减第0行，放在第1行，依此类推，首行全为0；
Mat cvdxMat(Mat a,int c)
{
	int rows = a.rows;
    int cols = a.cols;
	Mat out=Mat::zeros(rows, cols, CV_64FC1);
	for(int i = c; i < rows-1+c; i++)       
	{  
		// 取源图像的指针  
		const double* ptra = a.ptr<double>(i-c); 
		const double* ptrb = a.ptr<double>(i+1-c);
		// 将输出数据指针存放输出图像  
		double* ptrout = out.ptr<double>(i);  
		for(int j=0; j < cols; j++)  
		{  
			ptrout[j]= cv::saturate_cast<double>(ptrb[j]-ptra[j]);  
		}  
	} 
	return out;
}

//dy差分，c=0：第1列减第0列，放在第0列，依此类推，最后一列全为0；
//c=1：第1列减第0列，放在第1列，依此类推，首列全为0；
Mat cvdyMat(Mat a,int c)
{
	int rows = a.rows;
    int cols = a.cols;
	Mat out=Mat::zeros(rows, cols, CV_64FC1);
	for(int i = 0; i < rows; i++)       
	{  
		// 取源图像的指针   
		const double* ptra = a.ptr<double>(i);
		// 将输出数据指针存放输出图像  
		double* ptrout = out.ptr<double>(i);  
		for(int j=c; j < cols-1+c; j++)  
		{  
			ptrout[j]= cv::saturate_cast<double>(ptra[j+1-c]-ptra[j-c]);  
		}  
	} 
	return out;
}

////遍历解包裹
//Mat cvUnwrapMat(Mat a)
//{
//	int rows = a.rows;
//    int cols = a.cols;
//	for (int j=1;j<cols-1;j++)
//	{
//		double ptra =a.at<double>(1,j);
//		double ptrb =a.at<double>(1,j+1);
//		if (abs(ptra-ptrb)>=3.14)
//		{
//			if (ptra<ptrb)
//			{
//				for (int i=j+1;i<cols;i++)
//				{
//					double ptrc=a.at<double>(1,i);
//					a.at<double>(1,i)=ptrc-2*3.14159;
//				}
//			}
//			else
//			{
//				for (int i=j+1;i<cols;i++)
//				{
//					double ptrc=a.at<double>(1,i);
//					a.at<double>(1,i)=ptrc+2*3.14159;
//				}
//			}
//		}
//	}
//    for (int j=0;j<cols;j++)  
//    {
//        for (int i=0;i<rows-1;i++)  
//        {
//			double ptra=a.at<double>(i,j);
//			double ptrb=a.at<double>(i+1,j);      //修改灰度值  
//			if (abs(ptra-ptrb)>=3.14)
//			{
//			if (ptra<ptrb)
//			{
//				for (int x=i+1;x<rows;x++)
//				{
//					double ptrc=a.at<double>(x,j);
//					a.at<double>(x,j)=ptrc-2*3.14159;
//				}
//			}
//			else
//			{
//				for (int x=i+1;x<rows;x++)
//				{
//					double ptrc=a.at<double>(x,j);
//					a.at<double>(x,j)=ptrc+2*3.14159;
//				}
//			}
//		}
//        }  
//    }
//	return a;
//}

//全局解包裹
Mat cvUnwrap1Mat(Mat a)
{
	int rows = a.rows;
    int cols = a.cols;
	//离散余弦变换临时变量
	Mat temp1;
	temp1.create(rows, cols, CV_64FC1);
	//离散反余弦临时变量
	Mat temp2=Mat::zeros(rows, cols, CV_64FC1);
	//输出量
	Mat out;
	out.create(rows, cols, CV_64FC1);
	//置NaN为0
	 /*for (int i=0;i<rows;i++)  
    {
        for (int j=0;j<cols;j++)  
        {
			if (_isnan(a.at<double>(i,j))==1)a.at<double>(i,j)=0;
		}
	 }*/
	 //dx差分,dy差分，二阶偏导
	 Mat dx1=cvAngleMat(cvdxMat(a,0));
	 Mat dy1=cvAngleMat(cvdyMat(a,0));
	 Mat dx2=cvdxMat(dx1,1);
	 Mat dy2=cvdyMat(dy1,1);
	 Mat f=dx2+dy2;
	 //离散余弦变换
	 dct(f,temp1,CV_DXT_FORWARD);
	 for (int i=0;i<rows;i++)  
	 {
		 // 取源图像的指针   
		 const double* ptr1 = temp1.ptr<double>(i);
		 // 将输出数据指针存放输出图像  
		 double* ptr2 = temp2.ptr<double>(i);
        for (int j=0;j<cols;j++)  
        {
			double k1=2*cos((i+1)*3.14159/rows);
			double k2=2*cos((j+1)*3.14159/cols);
			double k=k1+k2-4;
			ptr2[j]=ptr1[j]/k;
		}
	 }
	 dct(temp2,out,CV_DXT_INVERSE);//离散反余弦
	 return out;
}






 int main()
{
	double start = static_cast<double>(getTickCount());
	std::ios::sync_with_stdio(false);
 	// 读入图片
 	Mat_<double> img1s=imread("D:\\1.bmp",0);
	Mat_<double> img2s=imread("D:\\2.bmp",0);
	Mat_<double> img3s=imread("D:\\3.bmp",0);
	Mat_<double> img4s=imread("D:\\4.bmp",0);
	Mat_<double> img5s=imread("D:\\5.bmp",0);
	Mat_<double> img6s=imread("D:\\6.bmp",0);
	Mat_<double> img7s=imread("D:\\7.bmp",0);
	Mat_<double> img8s=imread("D:\\8.bmp",0/*CV_LOAD_IMAGE_GRAYSCALE*/);
	//图片相减
	Mat_<double> img86s=img8s-img6s;
	Mat_<double> img57s=img5s-img7s;
	Mat_<double> img42s=img4s-img2s;
	Mat_<double> img13s=img1s-img3s;
	//求相位角
	Mat  d_phase2=cvAtan2Mat(img86s,img57s);
	Mat  d_phase1=cvAtan2Mat(img42s,img13s);
	Mat  d_phase=d_phase2-d_phase1;
	Mat  wrapped_phase=cvAngleMat(d_phase);
	//滤波
	wrapped_phase=cvFilMat(wrapped_phase);
	//解包裹
	Mat unwrapped_phase = cvUnwrap1Mat(wrapped_phase);
	Mat result;
	normalize(unwrapped_phase,result,0,255,CV_MINMAX);
	//imshow("a",unwrapped_phase);
	imwrite("D:\\12.bmp",result);
	double time = ((double)getTickCount() - start) / getTickFrequency();
	cout << "所用时间为：" << time << "秒" << endl;
	waitKey();
	system("pause");
	
 }




  /*img1s.convertTo(img1s, CV_64FC1);
	img2s.convertTo(img2s, CV_64FC1);
	img3s.convertTo(img3s, CV_64FC1);
	img4s.convertTo(img4s, CV_64FC1);
	img5s.convertTo(img5s, CV_64FC1);
	img6s.convertTo(img6s, CV_64FC1);
	img7s.convertTo(img7s, CV_64FC1);
	img8s.convertTo(img8s, CV_64FC1);*/
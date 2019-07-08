#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Math.h>
#include <armadillo>
#include <stdio.h>
#include <tchar.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#define CV_VERSION_ID       CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif
#pragma comment( lib, cvLIB("core") )
#pragma comment( lib, cvLIB("imgproc") )
#pragma comment( lib, cvLIB("highgui") )
#pragma comment( lib, cvLIB("flann") )
#pragma comment( lib, cvLIB("features2d") )
#pragma comment( lib, cvLIB("calib3d") )
#pragma comment( lib, cvLIB("gpu") )
#pragma comment( lib, cvLIB("legacy") )
#pragma comment( lib, cvLIB("ml") )
#pragma comment( lib, cvLIB("objdetect") )
#pragma comment( lib, cvLIB("ts") )
#pragma comment( lib, cvLIB("video") )
#pragma comment( lib, cvLIB("contrib") )
#pragma comment( lib, cvLIB("nonfree") )



using namespace cv;
using namespace std;

// 下标M.at<float>(i,j)  方法1-1  
cv::Mat inverseColor1(cv::Mat srcImage)  
{  
    cv::Mat tempImage = srcImage.clone();  
    int row = tempImage.rows;  
    int col = tempImage.cols;  
    // 分别对各个通道进行反色处理  
    for (int i = 0; i < row; i++)  
    {  
        for (int j = 0; j < col; j++)  
        {  
            tempImage.at<cv::Vec3b>(i, j)[0] = 255 - tempImage.at<cv::Vec3b>(i, j)[0];  
            tempImage.at<cv::Vec3b>(i, j)[1] = 255 - tempImage.at<cv::Vec3b>(i, j)[1];  
            tempImage.at<cv::Vec3b>(i, j)[2] = 255 - tempImage.at<cv::Vec3b>(i, j)[2];  
        }  
    }  
    return tempImage;  
}  
// 方法1-2 下标M::at<float>(i,j)    
cv::Mat inverseColor2(cv::Mat srcImage)  
{  
    cv::Mat tempImage = srcImage.clone();  
  
    int row = tempImage.rows;  
    // 图像像素行实际的宽度  
    int step = tempImage.step;  
    // 直接对像素进行反色处理  
    for (int i = 0; i < row; i++)  
    {  
        for (int j = 0; j < step; j++)  
        {  
            tempImage.at<uchar>(i, j) = 255 - tempImage.at<uchar>(i, j);  
        }  
    }     
    return tempImage;  
}  
// 方法2  指针遍历Mat::ptr  
cv::Mat inverseColor3(cv::Mat srcImage)  
{  
    cv::Mat tempImage = srcImage.clone();  
    int row = tempImage.rows;  
    // 将3通道转换为单通道  
    int nStep = tempImage.cols * tempImage.channels();  
    for(int i = 0; i < row; i++)   
    {  
        // 取源图像的指针  
        const uchar* pSrcData = srcImage.ptr<uchar>(i);  
        // 将输出数据指针存放输出图像  
        uchar* pResultData = tempImage.ptr<uchar>(i);  
        for(int j=0; j < nStep; j++)  
        {  
            pResultData[j]= cv::saturate_cast<uchar>(255 - pSrcData[j]);  
        }  
    }  
    return tempImage;  
}  
// 方法3 使用迭代器MatConstIterator  
cv::Mat inverseColor4(cv::Mat srcImage)  
{  
    cv::Mat tempImage = srcImage.clone();  
    // 初始化源图像迭代器  
    cv::MatConstIterator_<cv::Vec3b> srcIterStart  = srcImage.begin<cv::Vec3b>();  
    cv::MatConstIterator_<cv::Vec3b> srcIterEnd = srcImage.end<cv::Vec3b>();  
    // 初始化输出图像迭代器  
    cv::MatIterator_<cv::Vec3b> resIterStart = tempImage.begin<cv::Vec3b>();  
    cv::MatIterator_<cv::Vec3b> resIterEnd = tempImage.end<cv::Vec3b>();  
    // 遍历图像反色处理  
    while( srcIterStart != srcIterEnd )  
    {  
        (*resIterStart)[0] = 255 - (*srcIterStart)[0];  
        (*resIterStart)[1] = 255 - (*srcIterStart)[1];  
        (*resIterStart)[2] = 255 - (*srcIterStart)[2];  
        // 迭代器递增  
        srcIterStart++;  
        resIterStart++;  
    }  
    return tempImage;  
}  
  
// 方法4 改进的指针方法isContinuous  
cv::Mat inverseColor5(cv::Mat srcImage)  
{  
    int row = srcImage.rows;  
    int col = srcImage.cols;  
    cv::Mat tempImage = srcImage.clone();  
    // 判断是否是连续图像，即是否有像素填充  
    if( srcImage.isContinuous() && tempImage.isContinuous() )  
    {  
        row = 1;  
        col = col * srcImage.rows * srcImage.channels();  
    }  
    // 遍历图像的每个像素  
    for(int i = 0; i < row; i++)  
    {  
        const uchar* pSrcData = srcImage.ptr<uchar>(i);  
        uchar* pResultData = tempImage.ptr<uchar>(i);  
        for(int j = 0; j < col; j++)  
        {  
            *pResultData++ = 255 - *pSrcData++;  
        }  
    }  
    return tempImage;  
}  
// 方法5 LUT查表法  
cv::Mat inverseColor6(cv::Mat srcImage)  
{  
    int row = srcImage.rows;  
    int col = srcImage.cols;  
    cv::Mat tempImage = srcImage.clone();  
    // 建立LUT 反色table  
    uchar LutTable[256];  
    for (int i = 0; i < 256; ++i)  
        LutTable[i] = 255 - i;  
    cv::Mat lookUpTable(1, 256, CV_8U);  
    uchar* pData = lookUpTable.data;   
    // 建立映射表  
    for( int i = 0; i < 256; ++i)  
        pData[i] = LutTable[i];  
    // 应用索引表进行查找  
    cv::LUT(srcImage, lookUpTable, tempImage);  
    return tempImage;  
}  
  
int main()  
{  
    // 装载图像转为灰度图像  
    cv::Mat srcImage = cv::imread("d://flower3.jpg");  
    if (!srcImage.data)     
        return -1;  
    cv::imshow("srcImage", srcImage);  
  
    cv::Mat resultImg1,resultImg2,resultImg3;  
    cv::Mat resultImg4,resultImg5,resultImg6;  
  
    // 测试方法1-1  
    double tTime;  
    tTime = (double)getTickCount();  
    const int nTimes = 100;  
    for(int i  = 0;  i < nTimes; i++)  
    {  
        resultImg1 = inverseColor1(srcImage);  
    }  
    tTime= 1000*((double)getTickCount() - tTime)/getTickFrequency();  
    tTime /= nTimes;  
    std::cout << "test1-1: "  << tTime << std::endl;  
    // 测试方法1-2  
    tTime = (double)getTickCount();  
    for(int i  = 0;  i < nTimes; i++)  
    {  
        resultImg2 = inverseColor2(srcImage);  
    }  
    tTime= 1000*((double)getTickCount() - tTime)/getTickFrequency();  
    tTime /= nTimes;  
    std::cout << "test1-2: "  << tTime << std::endl;  
    // 测试方法2  
    tTime = (double)getTickCount();  
    for(int i  = 0;  i < nTimes; i++)  
    {  
        resultImg3 = inverseColor3(srcImage);  
    }  
    tTime= 1000*((double)getTickCount() - tTime)/getTickFrequency();  
    tTime /= nTimes;  
    std::cout << "test2: "  << tTime << std::endl;  
    // 测试方法3  
    tTime = (double)getTickCount();  
    for(int i  = 0;  i < nTimes; i++)  
    {  
        resultImg4 = inverseColor4(srcImage);  
    }  
    tTime= 1000*((double)getTickCount() - tTime)/getTickFrequency();  
    tTime /= nTimes;  
    std::cout << "test3: "  << tTime << std::endl;  
    // 测试方法4  
    tTime = (double)getTickCount();  
    for(int i  = 0;  i < nTimes; i++)  
    {  
        resultImg5 = inverseColor5(srcImage);  
    }  
    tTime= 1000*((double)getTickCount() - tTime)/getTickFrequency();  
    tTime /= nTimes;  
    std::cout << "test4: "  << tTime << std::endl;  
    // 测试方法5  
    tTime = (double)getTickCount();  
    for(int i  = 0;  i < nTimes; i++)  
    {  
        resultImg6 = inverseColor5(srcImage);  
    }  
    tTime= 1000*((double)getTickCount() - tTime)/getTickFrequency();  
    tTime /= nTimes;  
    std::cout << "test5: "  << tTime << std::endl;  
  
    cv::imshow("resultImg1", resultImg1);  
    cv::imshow("resultImg2", resultImg2);  
    cv::imshow("resultImg3", resultImg3);  
    cv::imshow("resultImg4", resultImg4);  
    cv::imshow("resultImg5", resultImg5);  
    cv::imshow("resultImg6", resultImg6);  
  
    cv::waitKey(0);  
    return 0;  
} 
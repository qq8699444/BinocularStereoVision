#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;

using namespace cv;

struct pointPair
{
    Point3f worldPt;
    Point2f imagePt;
};

int main(int argc, const char* argv[]) 
{
    std::string imgfilename = "IMG_20191106_214510.jpg";
    cv::Mat img;
    float ocx,ocy;
    img = cv::imread(imgfilename);

    if (img.empty())
    {
        cout <<"failed to read picture " << endl;
        return -1;
    }


    cout << "img row:" << img.rows << endl;
    cout << "img col:" << img.cols << endl;

    ocx = img.cols / 2;
    ocy = img.rows / 2;


    pointPair   ptPairs[] = {
        {Point3f(0,0,4.3f), Point2f(46,440)},        //第一个盒子左上角，上
        {Point3f(0,10.8,4.3f), Point2f(337,724)},    //第一个盒子左下角，上
        {Point3f(0,10.8f,0.f), Point2f(381,791)},    //第一个盒子左下角，下
        {Point3f(23.0f,0.f,4.3f), Point2f(552,110)}, // 第一个盒子右上角，上

        //第二个盒子
        {Point3f(4.8,10.8f,0.f), Point2f(526,667)},        // 左上,下
        {Point3f(4.8,10.8f,8.f), Point2f(495,486)},        // 左上,上
        {Point3f(4.8,10.8f + 10.4f,0.f), Point2f(879,953)},   // 左下,下
        {Point3f(4.8,10.8f + 10.4f,8.f), Point2f(1000,855)}, // 左下,上

        {Point3f(4.8 + 10.4f,10.8f,8.f), Point2f(805,208)}, // 右上,上
        {Point3f(4.8 + 10.4f,10.8f + 10.4f,8.f), Point2f(1242,434)}, // 右下,上
    };
    const int ptCnt = sizeof(ptPairs) / sizeof(ptPairs[0]);

    
    for (int ptIdx = 0; ptIdx < ptCnt; ptIdx++)
    {
        Point2f& dstpt = ptPairs[ptIdx].imagePt;
        circle(img, dstpt, 13, { 255,0,255 });
    }
    //cv::imshow("img", img);
    //cv::waitKey(0);

    

    //stage 1

    Eigen::Matrix<double, ptCnt, 7>    matrixA;
    Eigen::Matrix<double, ptCnt, 1>    matrixb;
    
    for (int ptIdx = 0; ptIdx < ptCnt; ptIdx++)
    {
        Point3f srcpt = ptPairs[ptIdx].worldPt;
        Point2f dstpt = ptPairs[ptIdx].imagePt - Point2f(ocx, ocy);


        matrixA(ptIdx, 0) = dstpt.y * srcpt.x;
        matrixA(ptIdx, 1) = dstpt.y * srcpt.y;
        matrixA(ptIdx, 2) = dstpt.y * srcpt.z;
        matrixA(ptIdx, 3) = dstpt.y;
        matrixA(ptIdx, 4) = -1 * dstpt.x * srcpt.x;
        matrixA(ptIdx, 5) = -1 * dstpt.x * srcpt.y;
        matrixA(ptIdx, 6) = -1 * dstpt.x * srcpt.z;
        matrixb(ptIdx, 0) = dstpt.x;
    }

    auto ATA = matrixA.transpose() * matrixA;
    cout << "ATA :" << ATA << endl;
    cout << "ATA det:" << ATA.determinant() << endl;
    
    Eigen::Matrix<double, 7, 1> result = (matrixA.transpose() * matrixA).inverse() * matrixA.transpose() * matrixb;
    cout << "result:" << result << endl;

    //stage 2
    float a5 = result(4,0);
    float a6 = result(5, 0);
    float a7 = result(6, 0);
    float ty2 = 1.0f/(a5 * a5 + a6 * a6 + a7 * a7);
    float absty = sqrtf(ty2);

    //stage 3
    {
        float r21 = a5 * absty;
        float r22 = a6 * absty;
        float r23 = a7 * absty;

        float a1 = result(0, 0);
        float a2 = result(1, 0);
        float a3 = result(2, 0);
        float a4 = result(3, 0);

        //sx must be positive
        float sx = sqrtf(powf(a1*absty,2) + powf(a2*absty,2) + powf(a3*absty,2));
        float r11 = a1 * absty/sx;
        float r12 = a2 * absty/sx;
        float r13 = a3 * absty/sx;
        float tx = a4 * absty / sx;

        Point3f& srcpt = ptPairs[5].worldPt;
        Point2f& dstpt = ptPairs[5].imagePt;

        float x = r11 * srcpt.x + r12 * srcpt.y + r13 * srcpt.z + tx;
        float y = r21 * srcpt.x + r22 * srcpt.y + r23 * srcpt.z + absty;

        cout << "x:" << x << ", y: " << y  << endl;
    }
    cv::imshow("img", img);
    cv::waitKey(0);
    
    return 0;
}

#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;

using namespace cv;

struct PointPair
{
    Point3f worldPt;
    Point2f imagePt;
};

struct  ExtrinsicParam
{
    float rotateMat[9];
    float transfromVec[3];
    
};

struct IntrinsicParam
{
    float u0;
    float v0;

    float k1;
    float sx;
    float f;
};

static PointPair   ptPairs[] = {
        {Point3f(0,0,4.3f), Point2f(46,440)},        //第一个盒子左上角，上
        {Point3f(0,10.8f,4.3f), Point2f(337,724)},    //第一个盒子左下角，上
        {Point3f(0,10.8f,0.f), Point2f(381,791)},    //第一个盒子左下角，下
        {Point3f(23.0f,0.f,4.3f), Point2f(552,110)}, // 第一个盒子右上角，上

        //第二个盒子
        {Point3f(4.8f,10.8f,0.f), Point2f(526,667)},        // 左上,下
        {Point3f(4.8f,10.8f,8.f), Point2f(495,486)},        // 左上,上
        {Point3f(4.8f,10.8f + 10.4f,0.f), Point2f(879,953)},   // 左下,下
        {Point3f(4.8f,10.8f + 10.4f,8.f), Point2f(1000,855)}, // 左下,上

        {Point3f(4.8f + 10.4f,10.8f,8.f), Point2f(805,208)}, // 右上,上
        {Point3f(4.8f + 10.4f,10.8f + 10.4f,8.f), Point2f(1242,434)}, // 右下,上
};
static const int ptCnt = sizeof(ptPairs) / sizeof(ptPairs[0]);

void calcExtrinsicMidParam(IntrinsicParam& intrinsicParam, ExtrinsicParam& extrinsicParam, float result[7])
{
    Eigen::Matrix<float, ptCnt, 7>    matrixA;
    Eigen::Matrix<float, ptCnt, 1>    matrixb;

    for (int ptIdx = 0; ptIdx < ptCnt; ptIdx++)
    {
        const Point3f& srcpt = ptPairs[ptIdx].worldPt;
        const Point2f& dstpt = ptPairs[ptIdx].imagePt;


        matrixA(ptIdx, 0) = dstpt.y * srcpt.x;
        matrixA(ptIdx, 1) = dstpt.y * srcpt.y;
        matrixA(ptIdx, 2) = dstpt.y * srcpt.z;
        matrixA(ptIdx, 3) = dstpt.y;
        matrixA(ptIdx, 4) = -1 * dstpt.x * srcpt.x;
        matrixA(ptIdx, 5) = -1 * dstpt.x * srcpt.y;
        matrixA(ptIdx, 6) = -1 * dstpt.x * srcpt.z;
        matrixb(ptIdx, 0) = dstpt.x;
    }

    //auto ATA = matrixA.transpose() * matrixA;
    //cout << "ATA :" << ATA << endl;
    //cout << "ATA det:" << ATA.determinant() << endl;

    Eigen::Matrix<float, 7, 1> resultx = (matrixA.transpose() * matrixA).inverse() * matrixA.transpose() * matrixb;
    cout << "result:" << resultx << endl;

    for (int i =0;i< 7;i++)
    {
        result[i] = resultx(i, 0);
    }
}


float judgeTy(float extrinsicMidParam[7])
{
    float a5 = extrinsicMidParam[4];
    float a6 = extrinsicMidParam[5];
    float a7 = extrinsicMidParam[6];
    float ty2 = 1.0f / (a5 * a5 + a6 * a6 + a7 * a7);
    float absty = sqrtf(ty2);

    float ty;

    float r21 = a5 * absty;
    float r22 = a6 * absty;
    float r23 = a7 * absty;

    float a1 = extrinsicMidParam[0];
    float a2 = extrinsicMidParam[1];
    float a3 = extrinsicMidParam[2];
    float a4 = extrinsicMidParam[3];

    //sx must be positive
    float sx = sqrtf(powf(a1*absty, 2) + powf(a2*absty, 2) + powf(a3*absty, 2));
    float r11 = a1 * absty / sx;
    float r12 = a2 * absty / sx;
    float r13 = a3 * absty / sx;
    float tx = a4 * absty / sx;


    // 选一个远离图像中心的点
    const Point3f& srcpt = ptPairs[6].worldPt;
    const Point2f& dstpt = ptPairs[6].imagePt;

    float x = r11 * srcpt.x + r12 * srcpt.y + r13 * srcpt.z + tx;
    float y = r21 * srcpt.x + r22 * srcpt.y + r23 * srcpt.z + absty;

    cout << "x:" << x << ", y: " << y << endl;

    if (dstpt.x * x > 0 && dstpt.y * y > 0)
    {
        ty = absty;
    }
    else if (dstpt.x * x < 0 && dstpt.y * y < 0)
    {
        ty = -absty;
    }
    else
    {
        cout << "wrong ...." << endl;
        return -1;
    }

    return ty;
}

void calcExtrinsicParam(IntrinsicParam& intrinsicParam, ExtrinsicParam&  extrinsicParam, float extrinsicMidParam[7])
{
    float ty = extrinsicParam.transfromVec[1];
    float sx;
    float r11, r12, r13;
    float r21, r22, r23;

    extrinsicParam.rotateMat[3] = r21 = extrinsicMidParam[4] * ty;
    extrinsicParam.rotateMat[4] = r22 = extrinsicMidParam[5] * ty;
    extrinsicParam.rotateMat[5] = r23 = extrinsicMidParam[6] * ty;

    float a1 = extrinsicMidParam[0];
    float a2 = extrinsicMidParam[1];
    float a3 = extrinsicMidParam[2];
    float a4 = extrinsicMidParam[3];

    //sx must be positive
    intrinsicParam.sx = sx = sqrtf(powf(a1*ty, 2) + powf(a2*ty, 2) + powf(a3*ty, 2));
    extrinsicParam.rotateMat[0] = r11 = a1 * ty / sx;
    extrinsicParam.rotateMat[1] = r12 =  a2 * ty / sx;
    extrinsicParam.rotateMat[2] = r13 = a3 * ty / sx;
    extrinsicParam.transfromVec[0] = a4 * ty / sx;


    extrinsicParam.rotateMat[6] = r12 * r23 - r13 * r22;
    extrinsicParam.rotateMat[7] = r13 * r21 - r11 * r23;
    extrinsicParam.rotateMat[8] = r11 * r22 - r12 * r21;
}


void estimateftz(IntrinsicParam& intrinsicParam, ExtrinsicParam&  extrinsicParam)
{
    const float r11 = extrinsicParam.rotateMat[0], r12 = extrinsicParam.rotateMat[1], r13 = extrinsicParam.rotateMat[2];
    const float r21 = extrinsicParam.rotateMat[3], r22 = extrinsicParam.rotateMat[4], r23 = extrinsicParam.rotateMat[5];
    const float r31 = extrinsicParam.rotateMat[6], r32 = extrinsicParam.rotateMat[7], r33 = extrinsicParam.rotateMat[8];
    const float tx = extrinsicParam.transfromVec[0], ty = extrinsicParam.transfromVec[1];

    Eigen::Matrix<float, ptCnt * 2, 2>    matrixA;
    Eigen::Matrix<float, ptCnt * 2, 1>    matrixb;

    for (int ptIdx = 0; ptIdx < ptCnt; ptIdx++)
    {
        const Point3f& worldpt = ptPairs[ptIdx].worldPt;
        const Point2f& imagept = ptPairs[ptIdx].imagePt;

        float xi = r11 * worldpt.x + r12 * worldpt.y + r13 * worldpt.z + tx;
        float yi = r21 * worldpt.x + r22 * worldpt.y + r23 * worldpt.z + ty;
        float wi = r31 * worldpt.x + r32 * worldpt.y + r33 * worldpt.z;

        matrixA(2 * ptIdx + 0, 0) = yi;
        matrixA(2 * ptIdx + 0, 1) = -imagept.y;
        matrixb(2 * ptIdx + 0, 0) = wi * imagept.y;

        matrixA(2 * ptIdx + 1, 0) = intrinsicParam.sx * xi;
        matrixA(2 * ptIdx + 1, 1) = -imagept.x;
        matrixb(2 * ptIdx + 1, 0) = wi * imagept.x;


    }

    auto ATA2 = matrixA.transpose() * matrixA;
    std::cout << "matrixA:" << matrixA << endl;
    std::cout << "ATA2 det:" << ATA2.determinant() << endl;

    Eigen::Matrix<float, 2, 1> resultX = (matrixA.transpose() * matrixA).inverse() * matrixA.transpose() * matrixb;
    std::cout << "resultX:" << resultX << endl;

    intrinsicParam.f = resultX(0, 0);
    extrinsicParam.transfromVec[2] = resultX(1, 0);

    std::cout << "f:" << intrinsicParam.f << ",tz:" << extrinsicParam.transfromVec[2] << endl;
}

static float totalAveSquareError(IntrinsicParam& intrinsicParam, ExtrinsicParam&  extrinsicParam)
{
    float error2 = 0;
    const float r11 = extrinsicParam.rotateMat[0], r12 = extrinsicParam.rotateMat[1], r13 = extrinsicParam.rotateMat[2];
    const float r21 = extrinsicParam.rotateMat[3], r22 = extrinsicParam.rotateMat[4], r23 = extrinsicParam.rotateMat[5];
    const float r31 = extrinsicParam.rotateMat[6], r32 = extrinsicParam.rotateMat[7], r33 = extrinsicParam.rotateMat[8];
    const float tx = extrinsicParam.transfromVec[0], ty = extrinsicParam.transfromVec[1],tz = extrinsicParam.transfromVec[2];
    const float sx = intrinsicParam.sx;
    const float f = intrinsicParam.f;
    const float k1 = intrinsicParam.k1;

    for (int i =0; i < ptCnt; i++)
    {
        const Point3f& worldpt = ptPairs[i].worldPt;
        const Point2f& imagept = ptPairs[i].imagePt;

        float xi = r11 * worldpt.x + r12 * worldpt.y + r13 * worldpt.z + tx;
        float yi = r21 * worldpt.x + r22 * worldpt.y + r23 * worldpt.z + ty;
        float wi = r31 * worldpt.x + r32 * worldpt.y + r33 * worldpt.z;

        float r2 = imagept.x *imagept.x + imagept.y *imagept.y;
        float trustx = sx * imagept.x*(1 + k1 * r2);
        float trusty = imagept.y*(1 + k1 * r2);
        float predictx = f * xi / (wi + tz);
        float predicty = f * yi / (wi + tz);
        //printf("x:[%f - %f],y:[%f - %f]\n", trustx, predictx, trusty, predicty);

        error2 += powf(trustx - predictx, 2) + powf(trusty-predicty,2);
    }

    return error2/ptCnt;
}

void calcDelta(IntrinsicParam& intrinsicParam, ExtrinsicParam&  extrinsicParam, float& deltaf, float& deltak1, float& deltatz)
{
    float deltafsum = 0;
    float deltak1sum = 0;
    float deltatzsum = 0;
     

    const float r11 = extrinsicParam.rotateMat[0], r12 = extrinsicParam.rotateMat[1], r13 = extrinsicParam.rotateMat[2];
    const float r21 = extrinsicParam.rotateMat[3], r22 = extrinsicParam.rotateMat[4], r23 = extrinsicParam.rotateMat[5];
    const float r31 = extrinsicParam.rotateMat[6], r32 = extrinsicParam.rotateMat[7], r33 = extrinsicParam.rotateMat[8];
    const float tx = extrinsicParam.transfromVec[0], ty = extrinsicParam.transfromVec[1], tz = extrinsicParam.transfromVec[2];
    const float sx = intrinsicParam.sx;
    const float f = intrinsicParam.f;
    const float k1 = intrinsicParam.k1;

    for (int i = 0; i < ptCnt; i++)
    {
        const Point3f& worldpt = ptPairs[i].worldPt;
        const Point2f& imagept = ptPairs[i].imagePt;

        float xi = r11 * worldpt.x + r12 * worldpt.y + r13 * worldpt.z + tx;
        float yi = r21 * worldpt.x + r22 * worldpt.y + r23 * worldpt.z + ty;
        float wi = r31 * worldpt.x + r32 * worldpt.y + r33 * worldpt.z;

        float r2 = imagept.x *imagept.x + imagept.y *imagept.y;
        float trustx = (1.0f/sx) * imagept.x*(1 + k1 * r2);
        float trusty = imagept.y*(1 + k1 * r2);
        float predictx = f * xi / (wi + tz);
        float predicty = f * yi / (wi + tz);
        //printf("x:[%f - %f],y:[%f - %f]\n", trustx, predictx, trusty, predicty);

        deltak1sum += (trusty - predicty) *imagept.y*r2 + (trustx - predictx)*(1.0f / sx)*imagept.x*r2;
        deltafsum += (trusty - predicty) *(-1) *yi / (wi + tz) + (trustx - predictx)*(-1)* xi / (wi + tz);
        deltatzsum += (trusty - predicty) * yi / powf(wi + tz, 2) + (trustx - predictx) * xi / powf(wi + tz, 2);
    }

    deltaf = deltafsum / ptCnt;
    deltak1 = deltak1sum / ptCnt;
    deltatz = deltatzsum / ptCnt;
}
void nonlinearOptimize(IntrinsicParam& intrinsicParam, ExtrinsicParam&  extrinsicParam)
{
    float& f = intrinsicParam.f;
    float& tz = extrinsicParam.transfromVec[2];
    float& k1 = intrinsicParam.k1 = 0.f;
    float baselr = 0.3f;    

    for (int i = 0;i < 3000; i++)
    {
        float lr = baselr * expf(-i / 1000.f);
        auto error2 = totalAveSquareError(intrinsicParam, extrinsicParam);
        std::cout << "ave error :" << error2 << ",lr:" << lr << endl;

        float deltaf, deltak1, deltatz;
        calcDelta(intrinsicParam, extrinsicParam, deltaf, deltak1, deltatz);        
        
        //update 
        f -= lr * deltaf;
        k1 -= lr * deltak1;
        tz -= lr * deltatz;
    }
}
int main(int argc, const char* argv[]) 
{
    ExtrinsicParam  extrinsicParam;
    IntrinsicParam  intrinsicParam;

    std::string imgfilename = "IMG_20191106_214510.jpg";
    cv::Mat img;
    img = cv::imread(imgfilename);

    if (img.empty())
    {
        cout <<"failed to read picture " << endl;
        return -1;
    }


    cout << "img row:" << img.rows << endl;
    cout << "img col:" << img.cols << endl;

    intrinsicParam.u0 = img.cols / 2.0f;
    intrinsicParam.v0 = img.rows / 2.0f;
    
    for (int ptIdx = 0; ptIdx < ptCnt; ptIdx++)
    {
        Point2f& dstpt = ptPairs[ptIdx].imagePt;
        circle(img, dstpt, 13, { 255,0,255 });
    }
    //cv::imshow("img", img);
    //cv::waitKey(0);

    //prepare data,squeeze x,y to [-1 ~ +1]
    for (int ptIdx = 0; ptIdx < ptCnt; ptIdx++)
    {
        Point2f& imagePt = ptPairs[ptIdx].imagePt;
        //dstpt -= Point2f(intrinsicParam.u0, intrinsicParam.v0);
        //dstpt /= Point2f(intrinsicParam.u0, intrinsicParam.v0);
        imagePt.x = (1.0f*imagePt.x - intrinsicParam.u0) / intrinsicParam.u0;
        imagePt.y = (1.0f*imagePt.y - intrinsicParam.v0) / intrinsicParam.v0;
    }

    //stage 1
    float extrinsicMidParam[7];
    calcExtrinsicMidParam(intrinsicParam,extrinsicParam, extrinsicMidParam);
    

    //stage 2
    extrinsicParam.transfromVec[1] = judgeTy(extrinsicMidParam);

    
    //stage 3
    calcExtrinsicParam(intrinsicParam,extrinsicParam, extrinsicMidParam);


    //
    printf("sx = %f\n", intrinsicParam.sx);

    //stage 4
    estimateftz(intrinsicParam,extrinsicParam);

    //stage 5
    nonlinearOptimize(intrinsicParam, extrinsicParam);
    cv::imshow("img", img);
    cv::waitKey(0);
    
    return 0;
}

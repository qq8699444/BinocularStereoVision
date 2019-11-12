#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <Eigen/SVD>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "parser.h"
#include "dltalgo.h"


using namespace std;
using namespace cv;



static const std::string chessBoardCorrdFile[] = {
    "data\\corners_1.dat",
    "data\\corners_2.dat",
    "data\\corners_3.dat",
    "data\\corners_4.dat",
    "data\\corners_5.dat",
};

static const int chessBoardImageCnt = sizeof(chessBoardCorrdFile) / sizeof(chessBoardCorrdFile[0]);

struct HomoMat
{
    double value[9];
};

struct Intrinsics
{
    double kx;  //fx
    double ks;
    double u0;  //cx

    //
    double ky;
    double v0;

};

void calcV(int i,int j, double h[9], double v[6])
{
    
    double H[3][3];
    memcpy(H, h, 9 * sizeof(double));
    v[0] = H[0][i] * H[0][j];                       //hi1hj1
    v[1] = H[0][i] * H[1][j] + H[1][i] * H[0][j];   //hi2*hj1 + hi1*hj1
    v[2] = H[1][i] * H[1][j];                       //hi2*hj2
    v[3] = H[2][i] * H[0][j] + H[0][i] * H[2][j];   //hi3*hj1 + hi1*hj3
    v[4] = H[2][i] * H[1][j] + H[1][i] * H[2][j];   //hi3*hj2 + hi2*hj3
    v[5] = H[2][i] * H[2][j];                       //hi3hj3
}

void calcBMat(vector<HomoMat> HMats, double B[6])
{
    const int chessBoardImageCnt = HMats.size();

    // V*b = 0;
    Eigen::MatrixXd  matrixV(chessBoardImageCnt * 2, 6);
    for (int i = 0; i < chessBoardImageCnt; i++)
    {
        HomoMat& mat = HMats[i];

        
        double v11[6];
        double v12[6];
        double v22[6];
        calcV(0, 0, mat.value, v11);
        calcV(0, 1, mat.value, v12);
        calcV(1, 1, mat.value, v22);

        //V[2 * img_idx, :] = v12
        // V[2 * img_idx + 1, :] = v11 - v22
        for (int k=0;k < 6;k++)
        {
            matrixV(2 * i,   k) = v12[k];
            matrixV(2 * i+1, k) = v11[k] - v22[k];
        }
    }

    cout << "matrixV:" << matrixV << endl;
    //svd decomposition 

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrixV, Eigen::ComputeThinU | Eigen::ComputeThinV);

    auto U = svd.matrixU();
    auto V = svd.matrixV();
    auto A = svd.singularValues();

    cout << "A:" << A << endl;
    float minSingular = A(0);
    int minIdx = 0;
    for (int i = 1; i < A.rows();i++)
    {
        if (A(i) < minSingular)
        {
            minSingular = A(i);
            minIdx = i;
        }
    }


    cout << "V:" << V << endl;
    for (int k = 0; k < 6; k++)
    {
        B[k] = V(k, minIdx) /*/ V(5, 5)*/;
        cout <<  B[k] << endl;
    }

}


void calcIntrinsicsFromBMat(double B[6], Intrinsics& intrinsic)
{
    double B11 = B[0];
    double B12 = B[1];
    double B22 = B[2];
    double B13 = B[3];
    double B23 = B[4];
    double B33 = B[5];

    intrinsic.v0 = (B12*B13 - B11 * B23) / (B11*B22 - B12 * B12);
    float lamda = B33 - (B13*B13 + intrinsic.v0 * (B12*B13 - B11 * B23)) / B11;
    intrinsic.kx = sqrtf(lamda / B11);
    intrinsic.ky = sqrtf(lamda*B11 / (B11*B22 - B12 * B12));
    intrinsic.u0 = -B13 * intrinsic.kx * intrinsic.kx / lamda;
    intrinsic.ks = -B12 * intrinsic.kx * intrinsic.kx * intrinsic.ky / lamda;
}

int main(int argc, const char* argv[])
{

    vector<HomoMat> HMats;
    std::vector<Point2d> worldpts = prasePointSFromFile("data\\corners_real.dat");
    
    for (int i = 0; i < chessBoardImageCnt; i++)
    {
        const std::string& filepath = chessBoardCorrdFile[i];
        cout << "file path : " << filepath << endl;
        std::vector<Point2d> imagepts = prasePointSFromFile(filepath); 
        HomoMat homoMat;
        computeHomography(worldpts, imagepts, homoMat.value);

        for (int i = 0; i < 9; i++)
        {
            cout << homoMat.value[i] << endl;
        }

        HMats.push_back(homoMat);
    }

    double B[6];
    calcBMat(HMats,B);

    Intrinsics  intrinsic;
    calcIntrinsicsFromBMat(B, intrinsic);
    cout << "fx:" << intrinsic.kx << ",fy:" << intrinsic.ky << endl;
    cout << "cx:" << intrinsic.u0 << ",cy:" << intrinsic.v0 << endl;
    cout << "ks:" << intrinsic.ks  << endl;

    return 0;
}
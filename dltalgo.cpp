#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

using namespace cv;

bool computeHomography(std::vector<Point2d>& srcPts, std::vector<Point2d> dstPts, double homograpy[9])
{
    //A*x=b
    Eigen::MatrixXd  matrixA(srcPts.size() * 2, 8);
    Eigen::MatrixXd  matrixb(srcPts.size() * 2, 1);

    int ptIdx = 0;
    for (int y = 0; y < srcPts.size(); y++)
    {

        Point2d& worldpt = srcPts[ptIdx];
        Point2d& imagept = dstPts[ptIdx];


        matrixA(2 * ptIdx, 0) = worldpt.x;
        matrixA(2 * ptIdx, 1) = worldpt.y;
        matrixA(2 * ptIdx, 2) = 1;
        matrixA(2 * ptIdx, 3) = 0;
        matrixA(2 * ptIdx, 4) = 0;
        matrixA(2 * ptIdx, 5) = 0;
        matrixA(2 * ptIdx, 6) = -1 * worldpt.x * imagept.x;
        matrixA(2 * ptIdx, 7) = -1 * worldpt.y * imagept.x;
        matrixb(2 * ptIdx, 0) = imagept.x;

        matrixA(2 * ptIdx + 1, 0) = 0;
        matrixA(2 * ptIdx + 1, 1) = 0;
        matrixA(2 * ptIdx + 1, 2) = 0;
        matrixA(2 * ptIdx + 1, 3) = worldpt.x;
        matrixA(2 * ptIdx + 1, 4) = worldpt.y;
        matrixA(2 * ptIdx + 1, 5) = 1;
        matrixA(2 * ptIdx + 1, 6) = -1 * worldpt.x * imagept.y;
        matrixA(2 * ptIdx + 1, 7) = -1 * worldpt.y * imagept.y;
        matrixb(2 * ptIdx + 1, 0) = imagept.y;

        ptIdx++;

    }

    Eigen::MatrixXd tmp = matrixA.transpose() * matrixA;
    if (tmp.determinant() < 10e-3)
    {
        return false;
    }
    //Eigen::Matrix<float, 8, 1>
    Eigen::MatrixXd result = (matrixA.transpose() * matrixA).inverse() * matrixA.transpose() * matrixb;
    //cout << "result:" << result << endl;

    for (int i = 0;i < 8;i++)
    {
        homograpy[i] = result(i, 0);
    }

    homograpy[8] = 1.0f;
    return false;
}
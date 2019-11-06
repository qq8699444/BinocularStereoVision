#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>


using namespace std;

using namespace cv;



int main(int argc, const char* argv[]) 
{
    std::string imgfilename = "left10.jpg";
    cv::Mat img;
    img = cv::imread(imgfilename);

    if (img.empty())
    {
        cout <<"failed to read picture " << endl;
        return -1;
    }


    cv::Mat gray;
    cvtColor(img,gray,COLOR_BGR2GRAY);

    Size board_size = Size(9, 6);

    vector< Point2f > corners;
    bool found = findChessboardCorners(gray,board_size,corners);
    if (found)
    {
        drawChessboardCorners(img,board_size,corners,true);
    }
    
    cout << "img row:" << img.rows << endl;
    cout << "img col:" << img.cols << endl;
    
    
    vector<Point2f> srcpts;
    vector<Point2f> dstpts;

    for (int y = 0; y < board_size.height; y++)
    {
        for (int x = 0;x < board_size.width; x++)
        {
            srcpts.push_back(Point2f(x,y));
        }
    }

    Mat homoMat =    findHomography(srcpts, corners);
    cout << "homoMat:" << homoMat << endl;


    Eigen::Matrix<double, 9*6*2, 8>    matrixA;
    Eigen::Matrix<double, 9 * 6 * 2, 1>    matrixb;

    int ptIdx = 0;
    for (int y = 0; y < board_size.height; y++)
    {
        for (int x = 0; x < board_size.width; x++)
        {
            Point2f srcpt = srcpts[ptIdx];
            Point2f dstpt = corners[ptIdx];


            matrixA(2 * ptIdx, 0) = srcpt.x;
            matrixA(2 * ptIdx, 1) = srcpt.y;
            matrixA(2 * ptIdx, 2) = 1;
            matrixA(2 * ptIdx, 3) = 0;
            matrixA(2 * ptIdx, 4) = 0;
            matrixA(2 * ptIdx, 5) = 0;
            matrixA(2 * ptIdx, 6) = -1 * srcpt.x * dstpt.x;
            matrixA(2 * ptIdx, 7) = -1 * srcpt.y * dstpt.x;
            matrixb(2 * ptIdx, 0) = dstpt.x;

            matrixA(2 * ptIdx + 1, 0) = 0;
            matrixA(2 * ptIdx + 1, 1) = 0;
            matrixA(2 * ptIdx + 1, 2) = 0;
            matrixA(2 * ptIdx + 1, 3) = srcpt.x;
            matrixA(2 * ptIdx + 1, 4) = srcpt.y;
            matrixA(2 * ptIdx + 1, 5) = 1;
            matrixA(2 * ptIdx + 1, 6) = -1 * srcpt.x * dstpt.y;
            matrixA(2 * ptIdx + 1, 7) = -1 * srcpt.y * dstpt.y;
            matrixb(2 * ptIdx + 1, 0) = dstpt.y;

            ptIdx++;
        }
    }

    
    Eigen::Matrix<double, 8, 1> result = (matrixA.transpose() * matrixA).inverse() * matrixA.transpose() * matrixb;
    cout << "result:" << result << endl;

    cv::imshow("img", img);
    cv::waitKey(0);
    return 0;
}

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
    cv::imshow("img",img);
    cv::waitKey(0);
    

    return 0;
}

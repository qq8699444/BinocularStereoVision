#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "parser.h"
#include "dltalgo.h"

using namespace std;

using namespace cv;



int main(int argc, const char* argv[]) 
{
    std::vector<Point2f> worldpts = prasePointSFromFile("data\\corners_real.dat");
    std::vector<Point2f> imagepts = prasePointSFromFile("data\\corners_1.dat");

    Mat homoMat =    findHomography(worldpts, imagepts);
    cout << "homoMat:" << homoMat << endl;

    float homoMat2[9];
    computeHomography(worldpts, imagepts, homoMat2);
    for (int i =0; i < 9;i++)
    {
        cout << homoMat2[i] << endl;
    }

    return 0;
}

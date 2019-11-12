#ifndef __PARSE_H_
#define __PARSE_H_
#include <iostream>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <Eigen/SVD>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

using namespace cv;

vector<std::string> stringSplit(const string& str,const char delim = ' ');
std::vector<Point2d>    prasePointSFromFile(const string& filename);



#endif
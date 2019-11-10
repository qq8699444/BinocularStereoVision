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


static const std::string chessBoardImages[] = {
    "1.jpg",
    "2.jpg",
    "3.jpg",
    "4.jpg",
    "5.jpg",
};

static const int chessBoardImageCnt = sizeof(chessBoardImages) / sizeof(chessBoardImages[0]);
static const Size board_size = Size(9, 6);
static const float worldUnit = 10.0f;//10cm

bool findPointPairs(const std::string& image, vector<PointPair>&   ptParis)
{
    cv::Mat img;
    img = cv::imread(image);

    if (img.empty())
    {
        cout << "failed to read picture " << endl;
        return false;
    }

    cv::Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    vector< Point2f > corners;
    bool found = findChessboardCorners(gray, board_size, corners);
    if (!found)
    {
        return false;
    }

    drawChessboardCorners(img, board_size, corners, true);
    cv::imshow("img", img);
    cv::waitKey(0);

    ptParis.clear();
    int cornerIdx = 0;
    for (int y = 0; y < board_size.height; y++)
    {
        for (int x = 0; x < board_size.width; x++)
        {
            PointPair   ptpair;
            ptpair.worldPt = Point3f(x*worldUnit, y*worldUnit,0);
            ptpair.imagePt = corners[cornerIdx];
            cornerIdx++;
            ptParis.push_back(ptpair);
        }
    }
    return true;
}

void calcHomography(const std::string& image)
{
    vector<PointPair>   ptParis;
    findPointPairs(image, ptParis);
}
int main(int argc, const char* argv[])
{
    vector<PointPair>   ptParis;

    for (int i =0; i < chessBoardImageCnt;i++)
    {
        const std::string& image = chessBoardImages[i];
        calcHomography(image);
    }
    return 0;
}
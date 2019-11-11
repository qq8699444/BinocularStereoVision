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

bool findPointPairs(const std::string& image, vector<PointPair>& ptParis)
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
    cv::waitKey(10);

    ptParis.clear();
    int cornerIdx = 0;
    for (int y = 0; y < board_size.height; y++)
    {
        for (int x = 0; x < board_size.width; x++)
        {
            PointPair ptpair;
            ptpair.worldPt = Point3f(x*worldUnit, y*worldUnit, 0);
            ptpair.imagePt = corners[cornerIdx];
            cornerIdx++;
            ptParis.push_back(ptpair);
        }
    }
    return true;
}

Mat calcHomography(const std::string& image)
{
    vector<PointPair> ptParis;
    findPointPairs(image, ptParis);

    vector<Point2f> worldpts;
    vector<Point2f> imagepts;

    for (int i = 0; i < ptParis.size(); i++)
    {
        const Point3f& worldpt = ptParis[i].worldPt;
        const Point2f& imagept = ptParis[i].imagePt;

        worldpts.push_back(Point2f(worldpt.x, worldpt.y));
        imagepts.push_back(imagept);
    }

    Mat homoMat = findHomography(worldpts, imagepts);
    cout << "homoMat:" << homoMat << endl;
    return homoMat;
}


void calcBMat(vector<Mat> HMats,float B[6])
{
    const int imageCnt = HMats.size();
    Eigen::Matrix<double, chessBoardImageCnt*2, 6>    matrixV;
    for (int i = 0; i < imageCnt; i++)
    {
        Mat& mat = HMats[i];
        //double h[6];
        double h11 = mat.at<double>(0, 0);
        double h12 = mat.at<double>(1, 0);
        double h13 = mat.at<double>(2, 0);


        double h21 = mat.at<double>(0, 1);
        double h22 = mat.at<double>(1, 1);
        double h23 = mat.at<double>(2, 1);

        // {hi1hj1, }
        double v11[6] = { h11*h11, h11*h12 + h12 * h11, h12*h12, h13*h11 + h11 * h13, h13*h12 + h12 * h13, h13*h13 };
        double v12[6] = { h11*h21, h11*h22 + h12 * h21, h12*h22, h13*h21 + h11 * h23, h13*h22 + h12 * h23, h13*h23 };
        double v22[6] = { h21*h21, h21*h22 + h22 * h21, h22*h22, h23*h21 + h21 * h23, h23*h22 + h22 * h23, h23*h23 };


        //V[2 * img_idx, :] = v12
        // V[2 * img_idx + 1, :] = v11 - v22
        for (int k=0;k < 6;k++)
        {
            matrixV(2 * i,   k) = v12[k];
            matrixV(2 * i+1, k) = v11[k] - v22[k];
        }
    }

    //svd decomposition 
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrixV, Eigen::ComputeThinU | Eigen::ComputeThinV);

    auto U = svd.matrixU();
    auto V = svd.matrixV();
    auto A = svd.singularValues();

    cout << "V:" << V << endl;
    for (int k = 0; k < 6; k++)
    {
        B[k] = V(5, k) / V(5, 5);
    }

    cout << "A:" << A << endl;
}
int main(int argc, const char* argv[])
{
    vector<Mat> HMats;

    for (int i = 0; i < chessBoardImageCnt; i++)
    {
        const std::string& image = chessBoardImages[i];
        Mat homoMat_i = calcHomography(image);
        HMats.push_back(homoMat_i);
    }

    float B[6];
    calcBMat(HMats,B);

    float B11 = B[0];
    float B12 = B[1];
    float B22 = B[2];
    float B13 = B[3];
    float B23 = B[4];
    float B33 = B[5];

    float cy = (B12*B13 - B11 * B23) / (B11*B22 - B12 * B12);
    float lamda = B33 - (B13*B13 + cy * (B12*B13 - B11 * B23)) / B11;
    float fx = sqrtf(lamda / B11);
    float fy = sqrtf(lamda*B11 / (B11*B22 - B12 * B12));
    float cx = -B13 * fx * fx / lamda;



    cout << "fx:"<< fx << ",fy:" << fy << endl;
    cout << "cx:" << cx << ",cy:" << cy << endl;
    return 0;
}
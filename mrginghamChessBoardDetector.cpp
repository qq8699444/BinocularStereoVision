#include <iostream>
#include <vector>



#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "point.hh"
#include "mrgingham.hh"

using namespace std;

using namespace cv;
using namespace mrgingham;


int main(int argc, const char* argv[])
{
    string imgFile = "left10.jpg";

    cv::Mat image = cv::imread(imgFile.c_str(), IMREAD_GRAYSCALE);
    if (image.empty())
    {
        fprintf(stderr, "Couldn't open image '%s'\n", imgFile.c_str());
        return -1;
    }

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(8);

    cv::equalizeHist(image, image);
    clahe->apply(image, image);

    std::vector<PointDouble> points_out;
    debug_sequence_t    seq;
    int found_pyramid_level = find_chessboard_from_image_array(points_out,
            NULL,
            image,
            0,
            false, seq,
            "111");
    bool result = (found_pyramid_level >= 0);
    

    return 0;
}

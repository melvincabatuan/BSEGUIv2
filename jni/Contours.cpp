#include "Contours.h"
#include <opencv2/imgproc.hpp>

using namespace mhealth;

void Contours::apply(cv::Mat &src, cv::Mat &dst)
{
        cv::RNG rng(12345); // for random color
        cv::Scalar color;

    // Convert to Gray
        cv::cvtColor(src, srcGray, cv::COLOR_BGR2GRAY);

    // Detect the edges
        cv::Canny(srcGray, imgIntermediate, 50, 100);

    // Find contours
        cv::findContours(imgIntermediate, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

    // Draw contours
     for( size_t i = 0; i < contours.size(); i++ )
            {
                if(contours[i].size() > 50){
                	// Randomize color
                    color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255), 255);
                    cv::drawContours( dst, contours, i, color, 2, 8, hierarchy, 0, cv::Point() );
                }
            }

}
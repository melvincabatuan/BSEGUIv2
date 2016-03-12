#ifndef MHEALTH_CONTOURS_H
#define MHEALTH_CONTOURS_H

#include <opencv2/core.hpp>

namespace mhealth {

    class Contours {
    public:
        void apply(cv::Mat &src, cv::Mat &dst);

    private:
        cv::Mat srcGray;
        cv::Mat imgIntermediate;
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
    };

} // namespace mhealth


#endif //MHEALTH_CONTOURS_H
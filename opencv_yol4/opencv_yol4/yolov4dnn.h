#ifndef YOLOV4DNN_H
#define YOLOV4DNN_H

#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cstdlib>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>



class Yolo4DNN
{
public:
    // colors for bounding boxes
    static constexpr float CONFIDENCE_THRESHOLD = 0;
    static constexpr float NMS_THRESHOLD = 0.4;
    static constexpr int NUM_CLASSES = 80;
    std::vector<std::string> class_names;
    cv::dnn::Net net;
    std::vector<cv::String> output_names;
    std::ifstream class_file;
    Yolo4DNN();
    Yolo4DNN(const char *weight, const char *config, const char *classes, cv::dnn::Backend backend, cv::dnn::Target target, const float confidence_threshold, const float NMS_threshold, const int num_classes);
    virtual ~Yolo4DNN();
    int yolo_inference(cv::Mat& frame, std::vector<int>& indices, std::vector<cv::Rect>& boxes, std::vector<float>& scores);

protected:

private:
    float CURR_CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD;
    float CURR_NMS_THRESHOLD = NMS_THRESHOLD;
    int CURR_NUM_CLASSES = NUM_CLASSES;


};


#endif // YOLOV4DNN_H

#include "yolov4dnn.h"

int main()
{
	Yolo4DNN yolo;
//    yolo.NUM_CLASSES

	cv::VideoCapture source("demo.mp4");

	cv::Mat frame;
    while (cv::waitKey(1) < 1)
    {
        {
            source >> frame;
            std::vector<int> indices[yolo.NUM_CLASSES];
            std::vector<cv::Rect> boxes[yolo.NUM_CLASSES];
            std::vector<float> scores[yolo.NUM_CLASSES];
            if (frame.empty())
            {
                cv::waitKey();
                break;
            }
           yolo.yolo_inference(frame, indices[0], boxes[0], scores[0]);
        }
        cv::namedWindow("output");
        cv::imshow("output", frame);
    }

	return 1;

}
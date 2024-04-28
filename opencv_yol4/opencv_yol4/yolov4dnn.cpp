#include "yolov4dnn.h"

// colors for bounding boxes
static const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};
static auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);


Yolo4DNN::Yolo4DNN() 
{
  net = cv::dnn::readNetFromDarknet("yolov7-tiny.cfg", "yolov7-tiny.weights");
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
  output_names = net.getUnconnectedOutLayersNames();

  class_file.open("classes.txt");
  if (!class_file)
  {
      std::cerr << "failed to open classes.txt\n";
  }
  std::string line;
  while (std::getline(class_file, line))
      class_names.push_back(line);
}
Yolo4DNN::Yolo4DNN(const char *weight, const char *config, const char *classes, cv::dnn::Backend backend, cv::dnn::Target target, const float confidence_threshold, const float NMS_threshold, const int num_classes)
{
    CURR_CONFIDENCE_THRESHOLD = confidence_threshold;
    CURR_NMS_THRESHOLD = NMS_threshold;
    CURR_NUM_CLASSES = num_classes;
    
    net = cv::dnn::readNetFromDarknet(config, weight);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    output_names = net.getUnconnectedOutLayersNames();

    class_file.open(classes);
    if (!class_file)
    {
        std::cerr << "failed to open classes.txt\n";
    }
    std::string line;
    while (std::getline(class_file, line))
        class_names.push_back(line);
}


Yolo4DNN::~Yolo4DNN()
{
}

int Yolo4DNN::yolo_inference(cv::Mat& frame, std::vector<int>& indices, std::vector<cv::Rect>& boxes, std::vector<float>& scores)
{
    std::vector<cv::Mat> detections;
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(608, 608), cv::Scalar(), true, false, CV_32F);
    net.setInput(blob);
    net.forward(detections, output_names);

    for (auto& output : detections)
    {
        const auto num_boxes = output.rows;
        for (int i = 0; i < num_boxes; i++)
        {
            auto x = output.at<float>(i, 0) * frame.cols;
            auto y = output.at<float>(i, 1) * frame.rows;
            auto width = output.at<float>(i, 2) * frame.cols;
            auto height = output.at<float>(i, 3) * frame.rows;
            cv::Rect rect(x - width / 2, y - height / 2, width, height);

            for (int c = 0; c < CURR_NUM_CLASSES; c++)
            {
                auto confidence = *output.ptr<float>(i, 5 + c);
                if (confidence >= CURR_CONFIDENCE_THRESHOLD)
                {
                    (&boxes)[c].push_back(rect);
                    (&scores)[c].push_back(confidence);
                }
            }
        }
    }

    for (int c = 0; c < CURR_NUM_CLASSES; c++)
        cv::dnn::NMSBoxes((&boxes)[c], (&scores)[c], 0.0, CURR_NMS_THRESHOLD, (&indices)[c]);

    for (int c = 0; c < CURR_NUM_CLASSES; c++)
    {
        for (size_t i = 0; i < (&indices)[c].size(); ++i)
        {
            const auto color = colors[c % NUM_COLORS];

            auto idx = (&indices)[c][i];
            const auto& rect = (&boxes)[c][idx];
            cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

            std::ostringstream label_ss;
            label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << (&scores)[c][idx];
            auto label = label_ss.str();

            int baseline;
            auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
            cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
            cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
        }
    }


    return 1;
}



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
#include "ThreadCam.h"

constexpr float CONFIDENCE_THRESHOLD = 0;
constexpr float NMS_THRESHOLD = 0.4;
constexpr int NUM_CLASSES = 80;

// colors for bounding boxes
const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

int main_old()
{
    std::vector<std::string> class_names;
    {
        std::ifstream class_file("classes.txt");
        if (!class_file)
        {
            std::cerr << "failed to open classes.txt\n";
            return 0;
        }

        std::string line;
        while (std::getline(class_file, line))
            class_names.push_back(line);
    }

//    cv::VideoCapture source("rtsp://yeokiwi:i9549pal@192.168.1.187:554/stream1");
//    cv::VideoCapture source("rtsp://127.0.0.1:8554/test");
    cv::VideoCapture source("demo.mp4");

    auto net = cv::dnn::readNetFromDarknet("yolov7-tiny.cfg", "yolov7-tiny.weights");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    auto output_names = net.getUnconnectedOutLayersNames();

    cv::Mat frame, blob;
    std::vector<cv::Mat> detections;
    while (cv::waitKey(1) < 1)
    {
        source >> frame;
        if (frame.empty())
        {
            cv::waitKey();
            break;
        }

        auto total_start = std::chrono::steady_clock::now();
        cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(608, 608), cv::Scalar(), true, false, CV_32F);
        net.setInput(blob);

        auto dnn_start = std::chrono::steady_clock::now();
        net.forward(detections, output_names);
        auto dnn_end = std::chrono::steady_clock::now();

        std::vector<int> indices[NUM_CLASSES];
        std::vector<cv::Rect> boxes[NUM_CLASSES];
        std::vector<float> scores[NUM_CLASSES];

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

                for (int c = 0; c < NUM_CLASSES; c++)
                {
                    auto confidence = *output.ptr<float>(i, 5 + c);
                    if (confidence >= CONFIDENCE_THRESHOLD)
                    {
                        boxes[c].push_back(rect);
                        scores[c].push_back(confidence);
                    }
                }
            }
        }

        for (int c = 0; c < NUM_CLASSES; c++)
            cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);

        for (int c = 0; c < NUM_CLASSES; c++)
        {
            for (size_t i = 0; i < indices[c].size(); ++i)
            {
                const auto color = colors[c % NUM_COLORS];

                auto idx = indices[c][i];
                const auto& rect = boxes[c][idx];
                cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

                std::ostringstream label_ss;
                label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];
                auto label = label_ss.str();

                int baseline;
                auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
                cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
                cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
            }
        }

        auto total_end = std::chrono::steady_clock::now();

        float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
        float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
        std::ostringstream stats_ss;
        stats_ss << std::fixed << std::setprecision(2);
        stats_ss << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
        auto stats = stats_ss.str();

        int baseline;
        auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
        cv::rectangle(frame, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(frame, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

        cv::namedWindow("output");
        cv::imshow("output", frame);
    }

    return 0;
}

std::vector<std::string> class_names;
auto net = cv::dnn::readNetFromDarknet("yolov7-tiny.cfg", "yolov7-tiny.weights");
auto output_names = net.getUnconnectedOutLayersNames();
auto net1 = cv::dnn::readNetFromDarknet("yolov7-tiny.cfg", "yolov7-tiny.weights");
auto output_names1 = net1.getUnconnectedOutLayersNames();

int init_yolo(void)
{
    {
        std::ifstream class_file("classes.txt");
        if (!class_file)
        {
            std::cerr << "failed to open classes.txt\n";
            return 0;
        }

        std::string line;
        while (std::getline(class_file, line))
            class_names.push_back(line);
    }
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    net1.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net1.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    // net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    // net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

}

int yolo_inference(cv::Mat& frame, std::vector<int> &indices, std::vector<cv::Rect> &boxes, std::vector<float> &scores)
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

            for (int c = 0; c < NUM_CLASSES; c++)
            {
                auto confidence = *output.ptr<float>(i, 5 + c);
                if (confidence >= CONFIDENCE_THRESHOLD)
                {
                    (&boxes)[c].push_back(rect);
                    (&scores)[c].push_back(confidence);
                }
            }
        }
    }

    for (int c = 0; c < NUM_CLASSES; c++)
        cv::dnn::NMSBoxes((&boxes)[c], (&scores)[c], 0.0, NMS_THRESHOLD, (&indices)[c]);

    for (int c = 0; c < NUM_CLASSES; c++)
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

}

int yolo_inference1(cv::Mat& frame, std::vector<int>& indices, std::vector<cv::Rect>& boxes, std::vector<float>& scores)
{
    std::vector<cv::Mat> detections;
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(608, 608), cv::Scalar(), true, false, CV_32F);
    net1.setInput(blob);
    net1.forward(detections, output_names1);

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

            for (int c = 0; c < NUM_CLASSES; c++)
            {
                auto confidence = *output.ptr<float>(i, 5 + c);
                if (confidence >= CONFIDENCE_THRESHOLD)
                {
                    (&boxes)[c].push_back(rect);
                    (&scores)[c].push_back(confidence);
                }
            }
        }
    }

    for (int c = 0; c < NUM_CLASSES; c++)
        cv::dnn::NMSBoxes((&boxes)[c], (&scores)[c], 0.0, NMS_THRESHOLD, (&indices)[c]);

    for (int c = 0; c < NUM_CLASSES; c++)
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

}



int main()
{
    init_yolo();

    cv::VideoCapture source1("517364228a.mp4");
    cv::VideoCapture source("demo.mp4");

    
    cv::Mat frame,frame1;
    while (cv::waitKey(1) < 1)
    {
        {
            source >> frame;
            std::vector<int> indices[NUM_CLASSES];
            std::vector<cv::Rect> boxes[NUM_CLASSES];
            std::vector<float> scores[NUM_CLASSES];
            if (frame.empty())
            {
                cv::waitKey();
                break;
            }
            yolo_inference(frame, indices[0], boxes[0], scores[0]);
        }
        cv::namedWindow("output");
        cv::imshow("output", frame);

        {
            source1 >> frame1;
            std::vector<int> indices[NUM_CLASSES];
            std::vector<cv::Rect> boxes[NUM_CLASSES];
            std::vector<float> scores[NUM_CLASSES];
            if (frame1.empty())
            {
                cv::waitKey();
                break;
            }
            yolo_inference1(frame1, indices[0], boxes[0], scores[0]);

        }



        cv::namedWindow("output1");
        cv::imshow("output1", frame1);
    }
}
int main1()
{
    cv::Mat Frame1, Frame2, FrameCon1;
    cv::Mat Frame3, Frame4, FrameCon2, FrameTotal;
    cv::Mat FrameE(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));  //a gray frame
    //
    /// In this example, 4 cameras are used. This is not necessary. 
    /// You can control any number, one, two or, for example, six. It does not matter.    
    //
    ThreadCam* Grb1, * Grb2, * Grb3, * Grb4;

    init_yolo();

    Grb1 = new ThreadCam();
    Grb2 = new ThreadCam();
    Grb3 = new ThreadCam();
    Grb4 = new ThreadCam();

    Grb1->Init("rtsp://yeokiwi:i9549pal@192.168.1.187:554/stream1");      //Camera 1
    Grb2->Init("rtsp://:8554/");      //Camera 2
 //   Grb2->Init("rtsp://yeokiwi:i9549pal@192.168.1.187:554/stream1");       //Camera 3
 //   Grb4->Init("rtsp://yeokiwi:i9549pal@192.168.1.187:554/stream1");       //Camera 4

    // use a gray frame to indicate an empty field (lost connection with camera, for instance)
    //
    ///  be sure every frame has the same size!
    ///  hconcat and vconcat expect equal width and height
    ///  otherwise OpenCV throws exceptions
    //
    while (true)
    {
        //get the new frame
        Grb1->GetFrame(Frame1);
        if (!Frame1.empty()) {
            std::vector<int> indices[NUM_CLASSES];
            std::vector<cv::Rect> boxes[NUM_CLASSES];
            std::vector<float> scores[NUM_CLASSES];
            yolo_inference(Frame1, indices[0], boxes[0], scores[0]);
        }
        else FrameE.copyTo(Frame1);
        //get the new frame
        Grb2->GetFrame(Frame2);
        if (!Frame2.empty()) {
            ///place here your time consuming algorithms
            std::vector<int> indices[NUM_CLASSES];
            std::vector<cv::Rect> boxes[NUM_CLASSES];
            std::vector<float> scores[NUM_CLASSES];
            yolo_inference(Frame2, indices[0], boxes[0], scores[0]);
        }
        else FrameE.copyTo(Frame2);
#if 0
        //get the new frame
        Grb3->GetFrame(Frame3);
        if (!Frame3.empty()) {
            ///place here your time consuming algorithms
        }
        else FrameE.copyTo(Frame3);

        //get the new frame
        Grb4->GetFrame(Frame4);
        if (!Frame4.empty()) {
            ///place here your time consuming algorithms
        }
        else FrameE.copyTo(Frame4);
        //concat the four frames
        cv::hconcat(Frame3, Frame4, FrameCon2);
        cv::vconcat(FrameCon1, FrameCon2, FrameTotal);
#endif
//        cv::hconcat(Frame1, Frame2, FrameCon1);
        //show the result
        cv::imshow("Camera", Frame2);
        char esc = cv::waitKey(5);
        if (esc == 27) break;
    }
    /// Gracefully, terminate the threads.
    Grb1->Quit();
    Grb2->Quit();
 //   Grb3->Quit();
//    Grb4->Quit();

    delete Grb1;
    delete Grb2;
//    delete Grb3;
//    delete Grb4;

    return 0;
}
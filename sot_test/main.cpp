#include <iostream>
#include <vector>
#include <cmath>
#include<iostream>
#include<thread>
#include<mutex>
#include <future>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

#include<condition_variable>
#include<chrono>

#include <bsnn_sot_tracker.h>  // 单目标跟踪
#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP    // 初始化LOG宏，有且只能使用一次


int main(int argc, const char *argv[])
{

    std::string model_path = "/home/root/obj_det/model/sot";

    Single_object_tracker single_object_tracker(2, 128, 4, 256, model_path);

    LOG(INFO) << "test sot...";
    
    cv::Mat z = cv::imread("128.jpg");
    cv::Mat x = cv::imread("256.jpg");

    // test sot.
    single_object_tracker.initialize(z);
    single_object_tracker.track(x);

    LOG(INFO) << "test sot done!!";


    return 0;
}
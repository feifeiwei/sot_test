#ifndef BSNN_SOT_TRACKER_H
#define BSNN_SOT_TRACKER_H
#include <iostream>
#include <vector>
#include <cmath>
#include <iostream>
#include <thread>
#include <mutex>
#include <tuple>
#include <future>
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <condition_variable>
#include <chrono>

#include <fstream>
#include <sstream>
#include "easylogging++.h"
#include <bsnn_user_api.h>


#define INSTANCE_TIMEOUT 5000

// --------------
//  跟踪前处理
// ------------------
float sample_target(const cv::Mat& im, cv::Mat& im_crop_pad, const cv::Rect& target_box, float search_area_factor, int output_wh);

// push data to bsnn
int get_input_data(const cv::Mat& sample, uint8_t* input_data, int img_h, int img_w);



// 单目标跟踪
class Single_object_tracker
{
public:
	Single_object_tracker(int tmp_factor, int tmp_size, int sch_factor, int sch_size, std::string& model_path);
	~Single_object_tracker();
	void initialize(const cv::Mat& image);
	cv::Mat get_z();
	void track(const cv::Mat& image);

	void postProcess(bsnn_user_output_t* p_u_output, cv::Mat& image);

	void clip_box(cv::Rect& box, const int H, const int W, const int margin = 10);
	cv::Rect map_box_back(const cv::Rect& pred_boxes, const float resize_factor);

private:
    int template_factor;
    int template_size;
    int search_factor;
    int search_size;
    // SOT network define.

    cv::Rect state;
    cv::Mat z_template;
    cv::Mat x_search;

    cv::Rect2f pred_bbox; // 预测框

    // bsnn network define.
    bsnn_user_instance_cfg_t u_ins_cfg; 
	bsnn_user_model_t u_model;
	bsnn_user_instance_t u_ins;
	bsnn_user_input_t input;
	bsnn_user_output_t u_output;
};




#endif 
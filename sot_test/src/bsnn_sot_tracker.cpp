#include "bsnn_sot_tracker.h"


// 辅助函数：递归遍历tuple并将其元素乘以两个参数的乘积
template <typename Tuple, size_t... Is, typename T, typename U>
void multiplyByTwoParametersHelper(Tuple& t, std::index_sequence<Is...>, const T& multiplier1, const U& multiplier2) {
    using expand_type = int[];
    (void)expand_type {
        0, ((std::get<Is>(t) *= multiplier1 * multiplier2), void(), 0)...
    };
}

// 外部接口函数：调用辅助函数以遍历tuple并将其元素乘以两个参数的乘积
template <typename... Args, typename T, typename U>
void multiplyByTwoParameters(std::tuple<Args...>& t, const T& multiplier1, const U& multiplier2) {
    multiplyByTwoParametersHelper(t, std::make_index_sequence<sizeof...(Args)>(), multiplier1, multiplier2);
}





int get_input_data(const cv::Mat& sample, uint8_t* input_data, int img_h, int img_w)
{
	// 分离通道送入
    // cv::Mat b(img_h, img_w, CV_8UC1);
    // cv::Mat g(img_h, img_w, CV_8UC1);
    // cv::Mat r(img_h, img_w, CV_8UC1);
    std::vector<cv::Mat> chanels;
    cv::split(sample, chanels);

    cv::Mat b = chanels.at(0);
    cv::Mat g = chanels.at(1);
    cv::Mat r = chanels.at(2);

    // test other input  --------------------------need to comment.
    // b.setTo(cv::Scalar(23));
    // g.setTo(cv::Scalar(44));
    // r.setTo(cv::Scalar(155));
        // 0.00045s
    for(int j=0; j<img_h; j++)
        memcpy(input_data+j*img_w, r.data+j*img_w, img_w);

    for(int j=0; j<img_h; j++)
        memcpy(input_data+j*img_w+img_h*img_w, g.data+j*img_w, img_w);
    
    for(int j=0; j<img_h; j++)
        memcpy(input_data+j*img_w+img_h*img_w*2, b.data+j*img_w, img_w);

    return 0;
}


// 单目标跟踪
Single_object_tracker::Single_object_tracker(int tmp_factor, int tmp_size, int sch_factor, int sch_size, std::string& model_path):
												template_factor(tmp_factor), template_size(tmp_size), 
											        search_factor(sch_factor), search_size(sch_size) 											   
{
	const char *path = model_path.c_str(); //"/home/root/obj_det/model/sot";
	LOG(INFO)  << "==>  init bsnn sot model path: " << path;
	int target_engine = 0;   // (1,4) (0,2)
	u_ins_cfg.mac_arr_num = 2;
	u_ins_cfg.queue_type = GET_IN_OUT_NON_BLOCKING;;
	u_ins_cfg.in_type = USER_ADDR;
	u_ins_cfg.input_bufferq_size = 1;
	u_ins_cfg.output_bufferq_size = 1;
	u_ins_cfg.in_out_timeout_period = INSTANCE_TIMEOUT;
	u_ins_cfg.batch_size = 1;

	// model initialization
	if (BSNN_SUCC != bsnn_model_load(&u_model, path))
	{
		LOG(ERROR) << "ERROR! Unable to load model";
		return;
	}

	if (bsnn_asic_type_is_b0())
	{
		u_ins_cfg.target_engine = (bsnn_target_engine_t) target_engine; //cfg.target_engine;
	}

	if (BSNN_SUCC != bsnn_instance_create(&u_model, &u_ins, &u_ins_cfg))
	{
		LOG(ERROR) << "sot model " << u_model.id << " instance " << u_ins.ins_id << " ERROR! Unable to create instance";
		return;
	}
	int input_ts_num = u_model.input_ts_num;
	int output_ts_num = u_model.output_ts_num;

	LOG(INFO)  << "SOT Model " << path << " on engine " << ((target_engine == BSNN_ENGINE_NET_2_MACA) ? "top" : "lite");
	LOG(INFO)  << "mac_arr_num: " << u_ins_cfg.mac_arr_num;
	LOG(INFO)  << "SOT input_ts_num " <<  input_ts_num; 
	LOG(INFO)  << "SOT output_ts_num " <<  output_ts_num; 
	LOG(INFO)  << "========";
};




Single_object_tracker::~Single_object_tracker() {};



void Single_object_tracker::initialize(const cv::Mat& image)
{

    z_template = image;
    LOG(INFO)  << "z_template local : " << z_template.size();
	LOG(INFO)  <<   "Single_object_tracker initialize successfully!";
}

void Single_object_tracker::postProcess(bsnn_user_output_t* p_u_output, cv::Mat& image)
{
	int8_t* out1 =  (int8_t*) p_u_output->ts_addr[0]; //  DWH 1, 16, 16, 1.  -> 16,16,16,1  
	int8_t* out2 =  (int8_t*) p_u_output->ts_addr[1]; //  /   2, 16, 16, 1.  -> 16,16,16,1  
	int8_t* out3 =  (int8_t*) p_u_output->ts_addr[2];

	int output_ts_num = p_u_output->output_ts_num;
	float ts_qnt_scale0 = p_u_output->ts_qnt_scale[0];
	float ts_qnt_scale1 = p_u_output->ts_qnt_scale[1];
	float ts_qnt_scale2 = p_u_output->ts_qnt_scale[2];

	float ts_deqnt_scale_per_channel0 = p_u_output->ts_deqnt_scale_per_channel[0][0];
	float ts_deqnt_scale_per_channel1 = p_u_output->ts_deqnt_scale_per_channel[1][0];
	float ts_deqnt_scale_per_channel2 = p_u_output->ts_deqnt_scale_per_channel[2][0];

	LOG(INFO)  << "【DEBUG】 output_ts_num = " <<output_ts_num;
	LOG(INFO)  << "【DEBUG】 ts_qnt_scale0 = " <<ts_qnt_scale0;
	LOG(INFO)  << "【DEBUG】 ts_qnt_scale1 = " <<ts_qnt_scale1;
	LOG(INFO)  << "【DEBUG】 ts_qnt_scale2 = " <<ts_qnt_scale2;
	LOG(INFO)  << "【DEBUG】 ts_deqnt_scale_per_channel0 = " <<ts_deqnt_scale_per_channel0;
	LOG(INFO)  << "【DEBUG】 ts_deqnt_scale_per_channel1 = " <<ts_deqnt_scale_per_channel1;
	LOG(INFO)  << "【DEBUG】 ts_deqnt_scale_per_channel2 = " <<ts_deqnt_scale_per_channel2;

	int fmp_size = 16;
	int pitch_channels = 16; // 内存对齐

	float output1[fmp_size][fmp_size][1] = {0};
	float output2[fmp_size][fmp_size][2] = {0};
	float output3[fmp_size][fmp_size][2] = {0};

	float* data = (float*) output1;
	for(int k=0; k<(fmp_size*fmp_size); k++)
	for(int j=0; j<1; j++)
	{
		*data++ = (float)*(out1 + (k*pitch_channels)+j) * ts_deqnt_scale_per_channel0;
		LOG(INFO)  << "raw conf: "<< +*(out1 + (k*pitch_channels)+j);
	}

	data = (float*) output2;
	for(int k=0; k<(fmp_size*fmp_size); k++)
	for(int j=0; j<2; j++)
	{
		*data++ =  (float)*(out2 + (k*pitch_channels)+j) * ts_deqnt_scale_per_channel1;
		LOG(INFO)  << "raw offset: "<< +*(out2 + (k*pitch_channels)+j);
	}

	data = (float*) output3;
	for(int k=0; k<(fmp_size*fmp_size); k++)
	for(int j=0; j<2; j++)
	{
		*data++ =  (float)*(out3 + (k*pitch_channels)+j) * ts_deqnt_scale_per_channel2;
	}

	// select maxium confidence score in feature score.
	float max_conf = 0.;
	int idx_x=0;
	int idx_y=0;

	float w = 0.;
	float h = 0.;
	float x_offset = 0.0;
	float y_offset = 0.0;

	for(int i=0; i<fmp_size;i++)
		for(int j=0; j<fmp_size; j++)
		{
			float conf = (output1[i][j][0]);
			if (conf > max_conf)
			{	idx_x = i;
				idx_y = j;
				max_conf = conf;
				LOG(INFO)  << "[> conf]: "<< conf << "，  idx x y = " << i << " " << j;
			}else
			{
				LOG(INFO)  << "[< conf]: "<< conf << "，  idx x y = " << i << " " << j;
			}
		}

	cv::Mat inputMat(16, 16, CV_32F, output1); 
	cv::Mat mask_inputMat = (inputMat < 0.01); // 小于0.01的置0
    inputMat.setTo(0, mask_inputMat);

	inputMat *= 255;

	cv::resize(inputMat, inputMat, cv::Size(256, 256), cv::INTER_NEAREST);
	inputMat.convertTo(inputMat, CV_8UC1);
	cv::cvtColor(inputMat, inputMat, cv::COLOR_GRAY2RGB);

	// 将两个图像混合
    double alpha = 0.5; // 调整alpha值以控制混合程度
    cv::addWeighted(inputMat, alpha, image, 1.0 - alpha, 0.0, image);
	cv::circle(image, cv::Point(idx_x*16, idx_y*16), 3, cv::Scalar(0,255,0), 3);
	cv::imwrite("debug_idx.jpg", image);

	w = output2[idx_x][idx_y][0];
	h = output2[idx_x][idx_y][1];

	x_offset = output3[idx_x][idx_y][0];
	y_offset = output3[idx_x][idx_y][1];
	
	LOG(INFO)  << "max conf: " << max_conf;
	LOG(INFO)  << "idx_x pos: " << idx_x;
	LOG(INFO)  << "idx_y pos: " << idx_y;
	LOG(INFO)  << "w: " << w;
	LOG(INFO)  << "h: " << h;
	LOG(INFO)  << "x_offset: " << x_offset;
	LOG(INFO)  << "y_offset: " << y_offset;

	//cx, cy, w, h
	float cx = ((float)idx_x + x_offset) / fmp_size;
	float cy = ((float)idx_y + y_offset) / fmp_size;

	// float cx = ((float)idx_x ) / fmp_size;
	// float cy = ((float)idx_y ) / fmp_size;

	pred_bbox.x = cx - w / 2;
	pred_bbox.y = cy - h / 2;
	pred_bbox.width = w;
	pred_bbox.height = h;
}


void Single_object_tracker::track(const cv::Mat& image)
{	


    x_search = image;
    // x_search.setTo(0);
    // cv::randu(x_search, cv::Scalar(0, 0, 0), cv::Scalar(256, 256, 256));
    LOG(INFO)  << "x_search local : " << x_search.size();

    if (BSNN_SUCC != bsnn_input_get(&u_ins, &input))
	{
		LOG(ERROR) << "failed to run bsnn_input_get()";
		return;
	}

	if (0!=get_input_data(z_template, (uint8_t*)input.frame_ts_usr_addr[0], template_size, template_size))
	{
		LOG(ERROR)  << "ERROR! get input data failed z_template" << std::endl;
		return;
	}

    if (0!=get_input_data(x_search, (uint8_t*)input.frame_ts_usr_addr[1], search_size, search_size))
	{
		LOG(ERROR)  << "ERROR! get input data failed x_search" << std::endl;
		return;
	}


	if (BSNN_SUCC != bsnn_input_set(&u_ins, &input))
	{
		LOG(ERROR) << "ERROR! failed to run bsnn_input_set()";
		// exit(-1);
		return;
	}

	if (bsnn_instance_run(&u_ins) != BSNN_SUCC)
	{
		LOG(ERROR)  << "model " << u_ins.model_id << ", instance " << u_ins.ins_id
			 << " ERROR! Unable to run" << std::endl;
		// exit(-1);
		return;
	}
	// get output
	int post_idx = BSNN_ERR_FAIL ; //-1; // what? BSNN_ERR_FAIL

	if (0 != bsnn_output_get(&u_ins, &u_output, &post_idx))
	{
		LOG(ERROR)  << "model " << u_ins.model_id << "instance " << u_ins.ins_id
			 << " ERROR! Unable to get output" << std::endl;
		return;
	}

	// post process
	this->postProcess(&u_output, x_search);


	if (BSNN_SUCC != bsnn_output_release(&u_ins, post_idx))
	{
		LOG(ERROR)  << "model " << u_ins.model_id << "instance " << u_ins.ins_id
			 << " ERROR! Unable to release output" << std::endl;
		return;
	}

	LOG(INFO)<< "Single_object_tracker track one step successfully.";
	exit(0);

}




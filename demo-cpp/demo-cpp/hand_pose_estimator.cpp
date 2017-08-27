/*************************************************************************
> File Name: output_pose.cpp
> Author: Guo Hengkai
> Description:
> Created Time: Sun 06 Nov 2016 10:47:29 AM CST
************************************************************************/
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

//#include <gflags/gflags.h>
//#include <glog/logging.h>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"

#include "hand_pose_estimator.h"


using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using std::string;
using std::vector;

HandPoseEstimator::HandPoseEstimator() {
	gpu = -1;	// cpu only
	fx = 463.889;
	fy = 463.889;
	ux = 320;
	uy = 240;
	lower = 1;
	upper = 650;
	cube_length.push_back(150);
	cube_length.push_back(150);
	cube_length.push_back(150);
	height = 96;
	width = 96;
	output_blob = "predict";
	weights = "models/model_icvl_ren_9x6x6.caffemodel";
	model = "models/deploy_icvl_ren_9x6x6.prototxt";
	init_model();
}

HandPoseEstimator::~HandPoseEstimator() {

}

int HandPoseEstimator::init_model() {
	// initialize model
	if (gpu < 0) {
		LOG(INFO) << "Using CPU";
		Caffe::set_mode(Caffe::CPU);
	}
	else {
		LOG(INFO) << "Using GPU " << gpu;
		Caffe::SetDevice(gpu);
		Caffe::set_mode(Caffe::GPU);
	}
	test_net = boost::shared_ptr<Net<float> >(new Net<float>(model, caffe::TEST));
	test_net->CopyTrainedLayersFrom(weights);
	return 1;
}

vector<float> HandPoseEstimator::predict(const cv::Mat& cv_img) {

	// get outputs and save
	const boost::shared_ptr<Blob<float> > blob =
		test_net->blob_by_name(output_blob);
	int batch_size = blob->num();
	int res_size = blob->count() / batch_size;
	LOG(INFO) << "batch size: " << batch_size;
	// pre processing
	vector<float> center;
	//cv::imshow("cv_img", cv_img);

	get_center(cv_img, center, lower, upper);
	cv::Mat crop = crop_image(cv_img, center, cube_length, fx, fy, height, width);
	//cv::imshow("crop", crop);

	//feed data
	boost::shared_ptr<Blob<float> > blob_data = test_net->blob_by_name("data");
	Mat2Blob(crop, blob_data);
	// forward
	test_net->Forward();
	const float* data = blob->cpu_data();

	vector<float> result;
	float c1 = center[0];
	float c2 = center[1];
	float c3 = center[2];
	std::cout << c1 << " " << c2 << " " << c3 << std::endl;

	for (int k = 0; k < res_size/3; ++k) {
		float u = *data;
		++data;
		float v = *data;
		++data;
		float d = *data;
		++data;
		// transform
		u = u * cube_length[0] * fabs(fx) / c3 + c1;
		v = v * cube_length[1] * fabs(fy) / c3 + c2;
		d = d * cube_length[2] + c3;
		//std::cout << u << " " << v << " " << d << std::endl;

		result.push_back(u);
		result.push_back(v);
		result.push_back(d);
	}

	return result;
}

void HandPoseEstimator::Mat2Blob(const cv::Mat &mat,
	boost::shared_ptr<Blob<float> > blob)
{
	assert(!mat.empty());
	float* blob_data = blob->mutable_cpu_data();

	int height = mat.rows;
	int width = mat.cols;
	int channel = 1;
	int top_index = 0;

	for (int c = 0; c < channel; ++c)
	{
		// cout << mat[c] << endl;
		for (int h = 0; h < height; ++h)
		{
			const float* ptr = mat.ptr<float>(h);
			for (int w = 0; w < width; ++w)
			{
				// int top_index = (c * height + h) * width + w;
				blob_data[top_index++] = ptr[w];
			}
		}
	}
}

void HandPoseEstimator::get_center(const cv::Mat& cv_img, vector<float>& center, int lower, int upper) {
	// TODO(guohengkai): remove the hard threshold if necessary 0 ~ 880
	center = vector<float>(3, 0);
	int count = 0;
	int min_val = INT_MAX;
	int max_val = INT_MIN;
	for (int r = 0; r < cv_img.rows; ++r) {
		const float* ptr = cv_img.ptr<float>(r);
		for (int c = 0; c < cv_img.cols; ++c) {
			//std::cout << ptr[c] << std::endl;
			if (ptr[c] <= upper && ptr[c] >= lower) {
				center[0] += c;
				center[1] += r;
				center[2] += ptr[c];
				++count;
			}
			/*
			if (int(ptr[c]) > 0)
			min_val = std::min(int(ptr[c]), min_val);
			max_val = std::max(int(ptr[c]), max_val);
			*/
		}
	}
	if (count) {
		for (int i = 0; i < 3; ++i) {
			center[i] /= count;
		}
		//std::cout << center[0] << " " << center[1] << " " << center[2] << std::endl;
	}
	else
	{
		center.clear();
		// LOG(INFO) << "max: " << max_val << ", min: " << min_val;
	}
}

cv::Mat HandPoseEstimator::crop_image(const cv::Mat& cv_img,
	const vector<float>& center, const vector<int>& cube_length,
	float fx, float fy, int height, int width) {
	float xstart = center[0] - cube_length[0] / center[2] * fabs(fx);
	float xend = center[0] + cube_length[0] / center[2] * fabs(fx);
	float ystart = center[1] - cube_length[1] / center[2] * fabs(fy);
	float yend = center[1] + cube_length[1] / center[2] * fabs(fy);
	float xscale = 2.0 / (xend - xstart);
	float yscale = 2.0 / (yend - ystart);
	//std::cout << "crop:" << xstart << " " << xend << " " << ystart << " " << yend << std::endl;
	//std::cout << "cube:" << cube_length[0] << " " << cube_length[1] << std::endl;

	vector<cv::Point2f> src, dst;
	src.push_back(cv::Point2f(xstart, ystart));
	dst.push_back(cv::Point2f(0, 0));
	src.push_back(cv::Point2f(xstart, yend));
	dst.push_back(cv::Point2f(0, height - 1));
	src.push_back(cv::Point2f(xend, ystart));
	dst.push_back(cv::Point2f(width - 1, 0));
	cv::Mat trans = cv::getAffineTransform(src, dst);
	cv::Mat res_img;
	cv::warpAffine(cv_img, res_img, trans, cv::Size(width, height),
		cv::INTER_LINEAR, cv::BORDER_CONSTANT, center[2] + cube_length[2]);
	res_img -= center[2];
	res_img = cv::max(res_img, -cube_length[2]);
	res_img = cv::min(res_img, cube_length[2]);
	res_img /= cube_length[2];
	return res_img;
}

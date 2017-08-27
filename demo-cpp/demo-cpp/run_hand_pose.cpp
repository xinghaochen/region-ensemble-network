// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

/////////////////////////////////////////////////////
// librealsense tutorial #1 - Accessing depth data //
/////////////////////////////////////////////////////

// First include the librealsense C++ header file
#include <librealsense/rs.hpp>
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "hand_pose_estimator.h"

int main() try
{
	// Create a context object. This object owns the handles to all connected realsense devices.
	rs::context ctx;
	printf("There are %d connected RealSense devices.\n", ctx.get_device_count());
	if (ctx.get_device_count() == 0) return EXIT_FAILURE;

	// This tutorial will access only a single device, but it is trivial to extend to multiple devices
	rs::device * dev = ctx.get_device(0);
	printf("\nUsing device 0, an %s\n", dev->get_name());
	printf("    Serial number: %s\n", dev->get_serial());
	printf("    Firmware version: %s\n", dev->get_firmware_version());

	// Configure depth to run at VGA resolution at 30 frames per second
	dev->enable_stream(rs::stream::depth, 640, 480, rs::format::z16, 30);

	dev->start();

	// Determine depth value corresponding to one meter
	const uint16_t one_meter = static_cast<uint16_t>(1.0f / dev->get_depth_scale());

	// caffe model
	HandPoseEstimator hpe;

	cv::Mat depth(480, 640, CV_32FC1);
	while (true)
	{
		// This call waits until a new coherent set of frames is available on a device
		// Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
		dev->wait_for_frames();

		// Retrieve depth data, which was previously configured as a 640 x 480 image of 16-bit depth values
		const uint16_t * depth_frame = reinterpret_cast<const uint16_t *>(dev->get_frame_data(rs::stream::depth));
		const uint16_t one_meter = static_cast<uint16_t>(1.0f / dev->get_depth_scale());

		// Print a simple text-based representation of the image, by breaking it into 10x20 pixel regions and and approximating the coverage of pixels within one meter
		for (int y = 0; y<480; ++y)
		{
			for (int x = 0; x<640; ++x)
			{
				float depth_val = *depth_frame++;
				if (depth_val > 0) {
					depth.at<float>(cv::Point(x, y)) = depth_val * dev->get_depth_scale() * 1000;
					//std::cout << depth_val * dev->get_depth_scale() * 1000 << std::endl;
				}
				else
					depth.at<float>(cv::Point(x, y)) = 10000;
			}
		}
		//std::cout << dev->get_depth_scale() << std::endl;
		cv::Mat dst;
		cv::flip(depth, dst, 1);
		//cv::imshow("dst", dst);
		vector<float> result = hpe.predict(dst);
		// show
		//for (int idx = 0; idx < result.size(); idx++)
		//	std::cout << result[idx] << std::endl;
		cv::Mat show(dst.clone());
		show.setTo(10000, show == 0);
		cv::threshold(show, show, 1000, 1000, cv::THRESH_TRUNC);

		double minVal, maxVal;
		cv::minMaxLoc(show, &minVal, &maxVal);
		show.convertTo(show, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

		cv::cvtColor(show, show, CV_GRAY2BGR);
		for (int i = 0; i < result.size()/3; ++i) {
			cv::circle(show, cv::Point2f(result[i*3], result[i*3+1]), 8,
				cv::Scalar(0, 0, 255), -1);
		}
		int joint_id_start[] = { 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14 };
		int  joint_id_end[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
		// draw bone
		for (int i = 0; i < 15; i++) {
			int k1 = joint_id_start[i];
			int k2 = joint_id_end[i];
			cv::line(show, cv::Point2f(result[k1 * 3], result[k1 * 3 + 1]),
				cv::Point2f(result[k2 * 3], result[k2 * 3 + 1]),
				cv::Scalar(0, 255, 0), 3);
		}
		cv::imshow("depth", show);
		cv::waitKey(1);
	}

	return EXIT_SUCCESS;
}
catch (const rs::error & e)
{
	// Method calls against librealsense objects may throw exceptions of type rs::error
	printf("rs::error was thrown when calling %s(%s):\n", e.get_failed_function().c_str(), e.get_failed_args().c_str());
	printf("    %s\n", e.what());
	return EXIT_FAILURE;
}

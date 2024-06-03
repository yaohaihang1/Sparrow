#ifndef MERGE_HDR_CUDA_CUH
#define MERGE_HDR_CUDA_CUH
#pragma once 
#include <device_launch_parameters.h>
//#include <device_functions.h>
#include <cuda_runtime.h>  
#include <cuda_texture_types.h>
#include <stdint.h>
#include <opencv2/core.hpp> 
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
__global__ void cuda_merge_hdr_6(const float*  depth_map_0,const float*  depth_map_1,const float*  depth_map_2,
	const float*  depth_map_3,const float*  depth_map_4,const float*  depth_map_5,
	const unsigned char* brightness_0,const unsigned char* brightness_1,const unsigned char* brightness_2,
	const unsigned char* brightness_3,const unsigned char* brightness_4,const unsigned char* brightness_5,
	uint32_t img_height, uint32_t img_width, float* const depth_map,unsigned char * const brightness);


__global__ void cuda_merge_hdr_5(const float*  depth_map_0,const float*  depth_map_1,const float*  depth_map_2,
	const float*  depth_map_3,const float*  depth_map_4,
	const unsigned char* brightness_0,const unsigned char* brightness_1,const unsigned char* brightness_2,
	const unsigned char* brightness_3,const unsigned char* brightness_4,
	uint32_t img_height, uint32_t img_width, float* const depth_map,unsigned char * const brightness);

__global__ void cuda_merge_hdr_4(const float*  depth_map_0,const float*  depth_map_1,const float*  depth_map_2,const float*  depth_map_3,
	const unsigned char* brightness_0,const unsigned char* brightness_1,const unsigned char* brightness_2,const unsigned char* brightness_3,
	uint32_t img_height, uint32_t img_width, float* const depth_map,unsigned char * const brightness);

__global__ void cuda_merge_hdr_3(const float*  depth_map_0,const float*  depth_map_1,const float*  depth_map_2,const unsigned char* brightness_0,const unsigned char* brightness_1,
	const unsigned char* brightness_2,uint32_t img_height, uint32_t img_width, float* const depth_map,unsigned char * const brightness);

__global__ void cuda_merge_hdr_2(const float*  depth_map_0,const float*  depth_map_1,const unsigned char* brightness_0,const unsigned char* brightness_1,
	uint32_t img_height, uint32_t img_width, float* const depth_map,unsigned char * const brightness);


__global__ void cuda_count_sum_pixel(const unsigned char* brightness,uint32_t img_height, uint32_t img_width, float* sum_pixels);

__global__ void cuda_count_sum_pixel_16(const unsigned short* brightness,uint32_t img_height, uint32_t img_width, float* sum_pixels);


void cuda_merge_hdr_6_cpu(cv::Mat depth_map_0, cv::Mat depth_map_1, cv::Mat depth_map_2,
	cv::Mat depth_map_3, cv::Mat depth_map_4, cv::Mat depth_map_5,
	cv::Mat brightness_0, cv::Mat brightness_1, cv::Mat brightness_2,
	cv::Mat brightness_3, cv::Mat brightness_4, cv::Mat brightness_5,
	int img_height, int img_width, cv::Mat& depth_map, cv::Mat& brightness);


void cuda_merge_hdr_5_cpu(cv::Mat depth_map_0, cv::Mat depth_map_1, cv::Mat depth_map_2,
	cv::Mat depth_map_3, cv::Mat depth_map_4, 
	cv::Mat brightness_0, cv::Mat brightness_1, cv::Mat brightness_2,
	cv::Mat brightness_3, cv::Mat brightness_4,
	int img_height, int img_width, cv::Mat& depth_map, cv::Mat& brightness);

void cuda_merge_hdr_4_cpu(cv::Mat depth_map_0, cv::Mat depth_map_1, cv::Mat depth_map_2,
	cv::Mat depth_map_3, 
	cv::Mat brightness_0, cv::Mat brightness_1, cv::Mat brightness_2,
	cv::Mat brightness_3,
	int img_height, int img_width, cv::Mat& depth_map, cv::Mat& brightness);

void cuda_merge_hdr_3_cpu(cv::Mat depth_map_0, cv::Mat depth_map_1, cv::Mat depth_map_2,
	cv::Mat brightness_0, cv::Mat brightness_1, cv::Mat brightness_2,
	int img_height, int img_width, cv::Mat& depth_map, cv::Mat& brightness);

void cuda_merge_hdr_2_cpu(cv::Mat depth_map_0, cv::Mat depth_map_1, cv::Mat brightness_0, cv::Mat brightness_1,
	int img_height, int img_width, cv::Mat& depth_map, cv::Mat& brightness);




#endif
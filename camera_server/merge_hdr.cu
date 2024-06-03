#include "merge_hdr.cuh"
#include<iostream>
#include <opencv2/core.hpp> 
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

__global__ void cuda_merge_hdr_6(const float*  depth_map_0,const float*  depth_map_1,const float*  depth_map_2,
	const float*  depth_map_3,const float*  depth_map_4,const float*  depth_map_5,
	const unsigned char* brightness_0,const unsigned char* brightness_1,const unsigned char* brightness_2,
	const unsigned char* brightness_3,const unsigned char* brightness_4,const unsigned char* brightness_5,
	uint32_t img_height, uint32_t img_width, float* const depth_map,unsigned char * const brightness)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int offset = idy * img_width + idx;

	if (idx < img_width && idy < img_height)
	{
 

		float pixel= 0;
		pixel +=  brightness_0[offset];
		pixel +=  brightness_1[offset];
		pixel +=  brightness_2[offset];
		pixel +=  brightness_3[offset];
		pixel +=  brightness_4[offset];
		pixel +=  brightness_5[offset];

		pixel/= 6.0;


		brightness[offset] = pixel;

		if(brightness_0[offset] < 255)
		{
			// brightness[offset] = brightness_0[offset];
			depth_map[offset] = depth_map_0[offset];
		}

		else if(brightness_1[offset] < 255)
		{
			// brightness[offset] = brightness_1[offset];
			depth_map[offset] = depth_map_1[offset];
		}
		else if(brightness_2[offset] < 255)
		{
			// brightness[offset] = brightness_1[offset];
			depth_map[offset] = depth_map_2[offset];
		}
		else if(brightness_3[offset] < 255)
		{
			// brightness[offset] = brightness_1[offset];
			depth_map[offset] = depth_map_3[offset];
		}
		else if(brightness_4[offset] < 255)
		{
			// brightness[offset] = brightness_1[offset];
			depth_map[offset] = depth_map_4[offset];
		}
		else
		{	
			// brightness[offset] = brightness_2[offset];
			depth_map[offset] = depth_map_5[offset];
		}
		//没有深度则用最亮的深度值
		if (depth_map[offset] <= 0)
		{
			depth_map[offset] = depth_map_0[offset];
		}
	}
}


void cuda_merge_hdr_6_cpu(cv::Mat depth_map_0, cv::Mat depth_map_1, cv::Mat depth_map_2,
	cv::Mat depth_map_3, cv::Mat depth_map_4, cv::Mat depth_map_5,
	cv::Mat brightness_0, cv::Mat brightness_1, cv::Mat brightness_2,
	cv::Mat brightness_3, cv::Mat brightness_4, cv::Mat brightness_5,
	int img_height, int img_width, cv::Mat& depth_map, cv::Mat& brightness)
{

	cv::Mat depth_map_0_now = depth_map_0.clone();
	cv::Mat depth_map_1_now = depth_map_1.clone();
	cv::Mat depth_map_2_now = depth_map_2.clone();
	cv::Mat depth_map_3_now = depth_map_3.clone();
	cv::Mat depth_map_4_now = depth_map_4.clone();
	cv::Mat depth_map_5_now = depth_map_5.clone();
	cv::Mat brightness_0_now = brightness_0.clone();
	cv::Mat brightness_1_now = brightness_1.clone();
	cv::Mat brightness_2_now = brightness_2.clone();
	cv::Mat brightness_3_now = brightness_3.clone();
	cv::Mat brightness_4_now = brightness_4.clone();
	cv::Mat brightness_5_now = brightness_5.clone();
	cv::Mat depthptr(img_height, img_width, CV_64F, cv::Scalar(0)); // 假设深度图是64位浮点数  
	cv::Mat brightptr(img_height, img_width, CV_8UC1, cv::Scalar(0));



	for (int i = 0; i < img_height; i++) {

		double* depth1 = depth_map_0_now.ptr<double>(i);
		double* depth2 = depth_map_1_now.ptr<double>(i);
		double* depth3 = depth_map_2_now.ptr<double>(i);
		double* depth4 = depth_map_3_now.ptr<double>(i);
		double* depth5 = depth_map_4_now.ptr<double>(i);
		double* depth6 = depth_map_5_now.ptr<double>(i);
		uchar* bright1 = brightness_0_now.ptr<uchar>(i);
		uchar* bright2 = brightness_1_now.ptr<uchar>(i);
		uchar* bright3 = brightness_2_now.ptr<uchar>(i);
		uchar* bright4 = brightness_3_now.ptr<uchar>(i);
		uchar* bright5 = brightness_4_now.ptr<uchar>(i);
		uchar* bright6 = brightness_5_now.ptr<uchar>(i);
		double* depth = depthptr.ptr<double>(i);
		uchar* bright = brightptr.ptr<uchar>(i);

		for (int j = 0; j < img_width; j++) {
			float pixel = 0;
			pixel += bright1[j];
			pixel += bright2[j];
			pixel += bright3[j];
			pixel += bright4[j];
			pixel += bright5[j];
			pixel += bright6[j];
			pixel /= 6;
			bright[j] = pixel;
			if (bright1[j] < 255) {
				depth[j] = depth1[j];
			}
			else if (bright2[j] < 255) {
				depth[j] = depth2[j];

			}
			else if (bright3[j] < 255) {
				depth[j] = depth3[j];

			}
			else if (bright4[j] < 255) {
				depth[j] = depth4[j];

			}
			else if (bright5[j] < 255) {
				depth[j] = depth5[j];

			}
			else {
				depth[j] = depth6[j];
			}
			if (depth[j] <= 0) {
				depth[j] = depth1[j];
			}
		}
	}

	depth_map = depthptr.clone();
	brightness = brightptr.clone();

}


__global__ void cuda_merge_hdr_5(const float*  depth_map_0,const float*  depth_map_1,const float*  depth_map_2,
	const float*  depth_map_3,const float*  depth_map_4,
	const unsigned char* brightness_0,const unsigned char* brightness_1,const unsigned char* brightness_2,
	const unsigned char* brightness_3,const unsigned char* brightness_4,
	uint32_t img_height, uint32_t img_width, float* const depth_map,unsigned char * const brightness)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int offset = idy * img_width + idx;

	if (idx < img_width && idy < img_height)
	{



		float pixel= 0;
		pixel +=  brightness_0[offset];
		pixel +=  brightness_1[offset];
		pixel +=  brightness_2[offset];
		pixel +=  brightness_3[offset];
		pixel +=  brightness_4[offset];

		pixel/= 5.0;


		brightness[offset] = pixel;

		if(brightness_0[offset] < 255)
		{
			// brightness[offset] = brightness_0[offset];
			depth_map[offset] = depth_map_0[offset];
		}

		else if(brightness_1[offset] < 255)
		{
			// brightness[offset] = brightness_1[offset];
			depth_map[offset] = depth_map_1[offset];
		}
		else if(brightness_2[offset] < 255)
		{
			// brightness[offset] = brightness_1[offset];
			depth_map[offset] = depth_map_2[offset];
		}
		else if(brightness_3[offset] < 255)
		{
			// brightness[offset] = brightness_1[offset];
			depth_map[offset] = depth_map_3[offset];
		}
		else
		{	
			// brightness[offset] = brightness_2[offset];
			depth_map[offset] = depth_map_4[offset];
		}
		//没有深度则用最亮的深度值
		if (depth_map[offset] <= 0)
		{
			depth_map[offset] = depth_map_0[offset];
		}
	}
}

void cuda_merge_hdr_5_cpu(cv::Mat depth_map_0, cv::Mat depth_map_1, cv::Mat depth_map_2,
	cv::Mat depth_map_3, cv::Mat depth_map_4,
	cv::Mat brightness_0, cv::Mat brightness_1, cv::Mat brightness_2,
	cv::Mat brightness_3, cv::Mat brightness_4,
	int img_height, int img_width, cv::Mat& depth_map, cv::Mat& brightness)
{
	

	cv::Mat depth_map_0_now = depth_map_0.clone();
	cv::Mat depth_map_1_now = depth_map_1.clone();
	cv::Mat depth_map_2_now = depth_map_2.clone();
	cv::Mat depth_map_3_now = depth_map_3.clone();
	cv::Mat depth_map_4_now = depth_map_4.clone();
	cv::Mat brightness_0_now = brightness_0.clone();
	cv::Mat brightness_1_now = brightness_1.clone();
	cv::Mat brightness_2_now = brightness_2.clone();
	cv::Mat brightness_3_now = brightness_3.clone();
	cv::Mat brightness_4_now = brightness_4.clone();
	cv::Mat depthptr(img_height, img_width, CV_64F, cv::Scalar(0)); // 假设深度图是64位浮点数  
	cv::Mat brightptr(img_height, img_width, CV_8UC1, cv::Scalar(0));


	for (int i = 0; i < img_height; i++) {

		double* depth1 = depth_map_0_now.ptr<double>(i);
		double* depth2 = depth_map_1_now.ptr<double>(i);
		double* depth3 = depth_map_2_now.ptr<double>(i);
		double* depth4 = depth_map_3_now.ptr<double>(i);
		double* depth5 = depth_map_4_now.ptr<double>(i);
		uchar* bright1 = brightness_0_now.ptr<uchar>(i);
		uchar* bright2 = brightness_1_now.ptr<uchar>(i);
		uchar* bright3 = brightness_2_now.ptr<uchar>(i);
		uchar* bright4 = brightness_3_now.ptr<uchar>(i);
		uchar* bright5 = brightness_4_now.ptr<uchar>(i);
		double* depth = depthptr.ptr<double>(i);
		uchar* bright = brightptr.ptr<uchar>(i);

		for (int j = 0; j < img_width; j++) {
			float pixel = 0;
			pixel += bright1[j];
			pixel += bright2[j];
			pixel += bright3[j];
			pixel += bright4[j];
			pixel += bright5[j];
			pixel /= 5;
			bright[j] = pixel;
			if (bright1[j] < 255) {
				depth[j] = depth1[j];
			}
			else if (bright2[j] < 255) {
				depth[j] = depth2[j];

			}
			else if (bright3[j] < 255) {
				depth[j] = depth3[j];

			}
			else if (bright4[j] < 255) {
				depth[j] = depth4[j];

			}
			else {
				depth[j] = depth5[j];
			}
			if (depth[j] <= 0) {
				depth[j] = depth1[j];
			}
		}
	}

	depth_map = depthptr.clone();
	brightness = brightptr.clone();
}



__global__ void cuda_merge_hdr_4(const float*  depth_map_0,const float*  depth_map_1,const float*  depth_map_2,const float*  depth_map_3,
	const unsigned char* brightness_0,const unsigned char* brightness_1,const unsigned char* brightness_2,const unsigned char* brightness_3,
	uint32_t img_height, uint32_t img_width, float* const depth_map,unsigned char * const brightness)
	{
		const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
		const unsigned int offset = idy * img_width + idx;
	
		if (idx < img_width && idy < img_height)
		{
	
	
	
			float pixel= 0;
			pixel +=  brightness_0[offset];
			pixel +=  brightness_1[offset];
			pixel +=  brightness_2[offset];
			pixel +=  brightness_3[offset];
	
			pixel/= 4.0;
	
	
			brightness[offset] = pixel;
	
			if(brightness_0[offset] < 255)
			{
				// brightness[offset] = brightness_0[offset];
				depth_map[offset] = depth_map_0[offset];
			}
	
			else if(brightness_1[offset] < 255)
			{
				// brightness[offset] = brightness_1[offset];
				depth_map[offset] = depth_map_1[offset];
			}
			else if(brightness_2[offset] < 255)
			{
				// brightness[offset] = brightness_1[offset];
				depth_map[offset] = depth_map_2[offset];
			}
			else
			{	
				// brightness[offset] = brightness_2[offset];
				depth_map[offset] = depth_map_3[offset];
			}
			//没有深度则用最亮的深度值
			if (depth_map[offset] <= 0)
			{
				depth_map[offset] = depth_map_0[offset];
			}
		}
	}


void cuda_merge_hdr_4_cpu(cv::Mat depth_map_0, cv::Mat depth_map_1, cv::Mat depth_map_2,
	cv::Mat depth_map_3,
	cv::Mat brightness_0, cv::Mat brightness_1, cv::Mat brightness_2,
	cv::Mat brightness_3,
	int img_height, int img_width, cv::Mat& depth_map, cv::Mat& brightness)
{

	cv::Mat depth_map_0_now = depth_map_0.clone();
	cv::Mat depth_map_1_now = depth_map_1.clone();
	cv::Mat depth_map_2_now = depth_map_2.clone();
	cv::Mat depth_map_3_now = depth_map_3.clone();
	cv::Mat brightness_0_now = brightness_0.clone();
	cv::Mat brightness_1_now = brightness_1.clone();
	cv::Mat brightness_2_now = brightness_2.clone();
	cv::Mat brightness_3_now = brightness_3.clone();
	cv::Mat depthptr(img_height, img_width, CV_64F, cv::Scalar(0)); // 假设深度图是64位浮点数  
	cv::Mat brightptr(img_height, img_width, CV_8UC1, cv::Scalar(0));



	for (int i = 0; i < img_height; i++) {

		double* depth1 = depth_map_0_now.ptr<double>(i);
		double* depth2 = depth_map_1_now.ptr<double>(i);
		double* depth3 = depth_map_2_now.ptr<double>(i);
		double* depth4 = depth_map_3_now.ptr<double>(i);
		uchar* bright1 = brightness_0_now.ptr<uchar>(i);
		uchar* bright2 = brightness_1_now.ptr<uchar>(i);
		uchar* bright3 = brightness_2_now.ptr<uchar>(i);
		uchar* bright4 = brightness_3_now.ptr<uchar>(i);
		double* depth = depthptr.ptr<double>(i);
		uchar* bright = brightptr.ptr<uchar>(i);

		for (int j = 0; j < img_width; j++) {
			float pixel = 0;
			pixel += bright1[j];
			pixel += bright2[j];
			pixel += bright3[j];
			pixel += bright4[j];
			pixel /= 4;
			bright[j] = pixel;
			if (bright1[j] < 255) {
				depth[j] = depth1[j];
			}
			else if (bright2[j] < 255) {
				depth[j] = depth2[j];

			}
			else if (bright3[j] < 255) {
				depth[j] = depth3[j];

			}
			else {
				depth[j] = depth4[j];
			}
			if (depth[j] <= 0) {
				depth[j] = depth1[j];
			}
		}
	}

	depth_map = depthptr.clone();
	brightness = brightptr.clone();


}




__global__ void cuda_merge_hdr_3(const float*  depth_map_0,const float*  depth_map_1,const float*  depth_map_2,const unsigned char* brightness_0,const unsigned char* brightness_1,
	const unsigned char* brightness_2,uint32_t img_height, uint32_t img_width, float* const depth_map,unsigned char * const brightness)
	{
		const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
		const unsigned int offset = idy * img_width + idx;
	
		if (idx < img_width && idy < img_height)
		{
	
	
	
			float pixel= 0;
			pixel +=  brightness_0[offset];
			pixel +=  brightness_1[offset];
			pixel +=  brightness_2[offset];
	
			pixel/= 3.0;
	
	
			brightness[offset] = pixel;
	
			if(brightness_0[offset] < 255)
			{
				// brightness[offset] = brightness_0[offset];
				depth_map[offset] = depth_map_0[offset];
			}
	
			else if(brightness_1[offset] < 255)
			{
				// brightness[offset] = brightness_1[offset];
				depth_map[offset] = depth_map_1[offset];
			}
			else
			{	
				// brightness[offset] = brightness_2[offset];
				depth_map[offset] = depth_map_2[offset];
			}
				//没有深度则用最亮的深度值
			if(depth_map[offset]<= 0)
			{
				depth_map[offset] = depth_map_0[offset];
			}
	
		}
	}

void cuda_merge_hdr_3_cpu(cv::Mat depth_map_0, cv::Mat depth_map_1, cv::Mat depth_map_2,
	cv::Mat brightness_0, cv::Mat brightness_1, cv::Mat brightness_2,
	int img_height, int img_width, cv::Mat& depth_map, cv::Mat& brightness)
{


	cv::Mat depth_map_0_now = depth_map_0.clone();
	cv::Mat depth_map_1_now = depth_map_1.clone();
	cv::Mat depth_map_2_now = depth_map_2.clone();
	cv::Mat brightness_0_now = brightness_0.clone();
	cv::Mat brightness_1_now = brightness_1.clone();
	cv::Mat brightness_2_now = brightness_2.clone();
	cv::Mat depthptr(img_height, img_width, CV_64F, cv::Scalar(0)); // 假设深度图是64位浮点数  
	cv::Mat brightptr(img_height, img_width, CV_8UC1, cv::Scalar(0));
	for (int i = 0; i < img_height; i++) {

		double* depth1 = depth_map_0_now.ptr<double>(i);
		double* depth2 = depth_map_1_now.ptr<double>(i);
		double* depth3 = depth_map_2_now.ptr<double>(i);
		uchar* bright1 = brightness_0_now.ptr<uchar>(i);
		uchar* bright2 = brightness_1_now.ptr<uchar>(i);
		uchar* bright3 = brightness_2_now.ptr<uchar>(i);
		double* depth = depthptr.ptr<double>(i);
		uchar* bright = brightptr.ptr<uchar>(i);

		for (int j = 0; j < img_width; j++) {
			float pixel = 0;
			pixel += bright1[j];
			pixel += bright2[j];
			pixel += bright3[j];
			pixel /= 3;
			bright[j] = pixel;
			if (bright1[j] < 255) {
				depth[j] = depth1[j];
			}
			else if(bright2[j]<255){
				depth[j] = depth2[j];

			}
			else {
				depth[j] = depth3[j];
			}
			if (depth[j] <= 0) {
				depth[j] = depth1[j];
			}
		}
	}
	depth_map = depthptr.clone();
	brightness = brightptr.clone();

}




__global__ void cuda_merge_hdr_2(const float*  depth_map_0,const float*  depth_map_1,const unsigned char* brightness_0,const unsigned char* brightness_1,
	uint32_t img_height, uint32_t img_width, float* const depth_map,unsigned char * const brightness)
	{
		const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
		const unsigned int offset = idy * img_width + idx;
	
		if (idx < img_width && idy < img_height)
		{
	
			float pixel= 0;
			pixel +=  brightness_0[offset];
			pixel +=  brightness_1[offset];
	
			pixel/= 2.0;
	
	
			brightness[offset] = pixel;
	
			if(brightness_0[offset] < 255)
			{
				// brightness[offset] = brightness_0[offset];
				depth_map[offset] = depth_map_0[offset];
			}
			else 
			{
				// brightness[offset] = brightness_1[offset];
				depth_map[offset] = depth_map_1[offset];
			}

			//没有深度则用最亮的深度值
			if(depth_map[offset]<= 0)
			{
				depth_map[offset] = depth_map_0[offset];
			}

		}
	}

void cuda_merge_hdr_2_cpu(cv::Mat depth_map_0, cv::Mat depth_map_1, cv::Mat brightness_0, cv::Mat brightness_1,
	int img_height, int img_width, cv::Mat& depth_map, cv::Mat& brightness)
{
	cv::Mat depth_map_0_now=depth_map_0.clone();
	cv::Mat depth_map_1_now = depth_map_1.clone();
	cv::Mat brightness_0_now = brightness_0.clone();
	cv::Mat brightness_1_now = brightness_1.clone();
	cv::Mat depthptr(img_height, img_width, CV_64F, cv::Scalar(0)); // 假设深度图是64位浮点数  
	cv::Mat brightptr(img_height, img_width, CV_8UC1, cv::Scalar(0));
	//cv::Mat brightptr = cv::Mat::zeros(img_height, img_width, CV_8U);
	
	for (int i = 0; i < img_height; i++) {
	
		double* depth1  = depth_map_0_now.ptr<double>(i);
		double* depth2 = depth_map_1_now.ptr<double>(i);
		uchar* bright1 = brightness_0_now.ptr<uchar>(i);
		uchar* bright2 = brightness_1_now.ptr<uchar>(i);

		double* depth = depthptr.ptr<double>(i);
		uchar* bright = brightptr.ptr<uchar>(i);

		for (int j = 0; j < img_width; j++) {
			float pixel = 0;
			pixel += bright1[j];
			pixel += bright2[j];
			pixel /= 2;
			bright[j] = pixel;
			if (bright1[j] < 255) {
				depth[j] = depth1[j];

		
			}
			else {
				depth[j] = depth2[j];
	
			}
			if (depth[j] <= 0) {
				depth[j] = depth1[j];
	
			}

		}

	}
	depth_map = depthptr.clone();
	brightness = brightptr.clone();


}




__global__ void cuda_count_sum_pixel(const unsigned char* brightness,uint32_t img_height, uint32_t img_width, float* sum_pixels)
{
		const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
		const unsigned int offset = idy * img_width + idx;
	
		if (idx < img_width && idy < img_height)
		{ 
			*sum_pixels +=  brightness[offset];  
		}
}


__global__ void cuda_count_sum_pixel_16(const unsigned short* brightness,uint32_t img_height, uint32_t img_width, float* sum_pixels)
{
		const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
		const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
		const unsigned int offset = idy * img_width + idx;
	
		if (idx < img_width && idy < img_height)
		{ 
			*sum_pixels +=  brightness[offset];  
		}
}
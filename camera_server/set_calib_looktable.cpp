#pragma once
#include<iostream>
#include <fstream>
#include <string.h>
#include <ctime>  
#include"LookupTableFunction.h"
#include"camera_param.h"
#include"set_calib_looktable.h"




int set_calib_looktable(float* calib_param_path, int width, int height)
{
	/*************************************************************************************************/
	struct CameraCalibParam calibration_param;
	std::ofstream file("calib_param.txt");
	//读取参数
	for (int i = 0; i < 40; i++) {
		((float*)(&calibration_param))[i]=calib_param_path[i];
		//std::cout << calib_param_path[i] << std::endl;
		file << calib_param_path[i] << std::endl;
	}
	file.close();
	//std::cout << "Read Param Finish！" << std::endl;

	//获取相机分辨率
	/*int width =1920;
	int height = 1200;*/
	//std::cout << "width: " << width << std::endl;
	//std::cout << "height: " << height << std::endl;

	//光机版本
	int version = 3010;

	//设置相机分辨率
	MiniLookupTableFunction minilooktable_machine;
	minilooktable_machine.setCameraResolution(width, height);


	//设置光机版本
	if (!minilooktable_machine.setProjectorVersion(version))
	{
		std::cout << "Set Projector Version failed!" << std::endl;
		std::cout << "version: " << version << std::endl;
		return -1;
	}

	//设置相机参数
	minilooktable_machine.setCalibData(calibration_param);
	cv::Mat xL_rotate_x;
	cv::Mat xL_rotate_y;
	cv::Mat rectify_R1;
	cv::Mat pattern_mapping;
	cv::Mat pattern_minimapping;


	//产生查找表
	//std::cout << "Start Generate LookTable Param" << std::endl;
	bool ok = minilooktable_machine.generateBigLookTable(xL_rotate_x, xL_rotate_y, rectify_R1, pattern_mapping, pattern_minimapping);
	//std::cout << "Finished Generate LookTable Param: " << ok << std::endl;

	//转类型
	xL_rotate_x.convertTo(xL_rotate_x, CV_32F);
	xL_rotate_y.convertTo(xL_rotate_y, CV_32F);
	rectify_R1.convertTo(rectify_R1, CV_32F);
	pattern_mapping.convertTo(pattern_mapping, CV_32F);
	pattern_minimapping.convertTo(pattern_minimapping, CV_32F);

	cv::Mat filling_map(pattern_mapping.size(), CV_8U, cv::Scalar(0));

	for (int i = 0; i < pattern_mapping.rows; i++)
	{
		for (int j = 0; j < pattern_mapping.cols; j++)
		{
			if (pattern_mapping.at<float>(i, j) != -2.)
			{
				filling_map.at<uchar>(i, j) = 255;
			}
		}
	}


	//保存图片查看
	//std::cout << "filling_map.rows: " << filling_map.rows << std::endl;
	//std::cout << "filling_map.cols: " << filling_map.cols << std::endl;
	cv::imwrite("pattern_mapping.bmp", filling_map);
	cv::imwrite("pattern_mapping.tiff", pattern_mapping);
	
	
	//保存5个bin文件
	LookupTableFunction lookup_table_machine_;
	lookup_table_machine_.saveBinMappingFloat("combine_xL_rotate_x_cam1_iter.bin", xL_rotate_x);
	lookup_table_machine_.saveBinMappingFloat("combine_xL_rotate_y_cam1_iter.bin", xL_rotate_y);
	lookup_table_machine_.saveBinMappingFloat("R1.bin", rectify_R1);
	lookup_table_machine_.saveBinMappingFloat("single_pattern_mapping.bin", pattern_mapping);
	lookup_table_machine_.saveBinMappingFloat("single_pattern_minimapping.bin", pattern_minimapping);

	return 1;
}

//
//int main() {
//	clock_t start = clock();
//
//	const char* calib_param_path = "param.txt";
//	int ret = 0;
//	ret = set_calib_looktable(calib_param_path);
//
//	clock_t end = clock();
//	double durationInSeconds = double(end - start) / CLOCKS_PER_SEC;
//	std::cout << "程序执行时间：" << durationInSeconds << " 秒" << std::endl;
//
//
//
//}
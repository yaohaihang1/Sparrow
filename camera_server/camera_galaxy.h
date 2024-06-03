#pragma once
#include "GxIAPI.h"
#include "camera.h"
#include "lightcrafter3010.h"
#include <chrono>         // std::chrono::milliseconds
#include <thread>         // std::thread
#include <mutex>          // std::timed_mutex

class CameraGalaxy : public Camera, public LightCrafter3010
{
public:
	CameraGalaxy();
	~CameraGalaxy();

	bool openCamera();
	bool closeCamera();

	bool switchToInternalTriggerMode();
	bool switchToExternalTriggerMode();

	bool getExposure(double& val);
	bool setExposure(double val);

	bool getGain(double& value);
	bool setGain(double value);

	bool trigger_software();
	bool grap(unsigned char* buf);
	bool grap(unsigned short* buf);

	bool setPixelFormat(int val);

	bool getPixelFormat(int& val);

	bool getMinExposure(float& val);


	bool streamOn();
	bool streamOff();

	void LINE0_IN();
	void LINE1_OUT(bool xOn);
	void SDA_OUTPUT();
	void SDA_ON();
	void SDA_OFF();
	void SDA_INPUT();
	bool SDA_READ();
	void SCL_OUTPUT();
	void SCL_ON_OFF();
	void SCL_ON();
	void SCL_OFF();

private:
	//void streamOffThread();
	void trigger_one_sequence();

	void trigger_hdr_list_flush();

	size_t read(char inner_reg, void* buffer, size_t buffer_size);
	size_t write(char inner_reg, void* buffer, size_t buffer_size);

private:
	float min_camera_exposure_;
	float camera_exposure;
	GX_DEV_HANDLE hDevice_;
	GX_FRAME_DATA  pFrameData;
	std::mutex operate_mutex_;
	bool line_1_inited_;

	int pixel_format_;
};
#pragma once
#include<iostream>
#include "xema_enums.h"


class Camera
{
public:
	Camera() {};
	~Camera() {};

	virtual bool openCamera() { return 1; };

	virtual bool closeCamera() { return 1; };
	
	virtual bool switchToInternalTriggerMode() { return 1; };

	virtual bool switchToExternalTriggerMode() { return 1; };

	virtual bool getExposure(double &val){ return 1; };
	virtual bool setExposure(double val){ return 1; };
    
	virtual bool getGain(double &val){ return 1; };
	virtual bool setGain(double val){ return 1; };
	  
	virtual bool streamOn(){ return 1; };
	virtual bool streamOff(){ return 1; };

    virtual bool trigger_software(){ return 1; };

    virtual bool grap(unsigned char* buf){ return 1; };

    virtual bool grap(unsigned short* buf){ return 1; };

	virtual bool setPixelFormat(int val){ return 1; };

	virtual bool getPixelFormat(int &val){ return 1; };


	bool getImageSize(int &width,int &height)
	{
		width = image_width_;
		height = image_height_;

		return true;
	}
	
	bool getMinExposure(float& val) { val = 10000; return true; };

	bool getPixelType(XemaPixelType& type) { type = XemaPixelType::Mono;  return 1; };

protected:
 
	bool camera_opened_state_; 
 

	long long int image_width_;
	long long int image_height_;

	float min_camera_exposure_; 
	float max_camera_exposure_; 
	
	bool stream_off_flag_;
	bool trigger_on_flag_;

	XemaPixelType pixel_type_;
};


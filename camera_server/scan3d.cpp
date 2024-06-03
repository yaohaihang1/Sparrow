#pragma once
#include "scan3d.h"
#include "easylogging++.h"
// #include "encode_cuda.cuh" 
#include "LookupTableFunction.h"  
#include "protocol.h"
#include "management.cuh"   
#include <opencv2/photo.hpp>
 
#include "management.cuh"  
#include "triangulation.h"

 
Scan3D::Scan3D()
{
    max_camera_exposure_ = 28000;
    min_camera_exposure_ = 1000;

    led_current_ = 1023;
    camera_exposure_ = 12000; 
    camera_gain_ = 0;

    generate_brightness_model_ = 1;
    generate_brightness_exposure_ = 12000;

    camera_opened_flag_ = false;

    
    fisher_confidence_val_ = -50;

    patterns_sets_num_ = 0;
}

Scan3D::~Scan3D()
{

}

int Scan3D::init()
{
    int ret = DF_SUCCESS;



    //相机初始
    camera_ = new CameraGalaxy(); 
    lc3010_ = camera_;
    


    if(!camera_->openCamera())
    { 
        LOG(INFO)<<"Open Galaxy Camera Error!";
        delete camera_;
    }
    else
    {
        LOG(INFO)<<"Open Galaxy Camera:";
        camera_opened_flag_ = true;
    }
    camera_->switchToExternalTriggerMode();    
    LOG(INFO)<<"switchToExternalTriggerMode!"; 

    camera_->getImageSize(image_width_,image_height_);

    XemaPixelType type;
    camera_->getPixelType(type);

    LOG(INFO)<<"PixelType: "<<(int)type;

    float min_exposure = 0;
    camera_->getMinExposure(min_exposure);
    LOG(INFO)<<"scan3d min_exposure: "<<min_exposure; 
    lc3010_->set_camera_min_exposure(min_exposure);

    min_camera_exposure_mono8_ = min_exposure;
    min_camera_exposure_mono12_ = min_exposure*2;

    buff_brightness_ = new unsigned char[image_width_*image_height_];
    buff_depth_ = new float[image_width_*image_height_];
    buff_pointcloud_ = new float[3*image_width_*image_height_];
 
    /*****************************************************************************************/
        //光机初始化
    //LOG(INFO) << __LINE__;
    ret = lc3010_->init();

    if (DF_SUCCESS != ret)
    {
        LOG(ERROR) << "lc3010 init FAILED; CODE : " << ret;
    }

    ret = lc3010_->read_patterns_sets_num(patterns_sets_num_);

    if (DF_SUCCESS != ret)
    {
        LOG(ERROR) << "read patterns sets FAILED; CODE : " << ret;
    }
    LOG(INFO) << "pattern_sets_num: " << patterns_sets_num_;


    lc3010_->enable_solid_field();

    LOG(INFO) << "lc3010 init";

    lc3010_->read_dmd_device_id(projector_version_);



    if(!setProjectorVersion(projector_version_))
    {
        LOG(INFO)<<"set Camera Version Failed!"; 
    }

    /**********************************************************************************************/

    if(0!= image_width_ && 0!= image_height_)
    {

        cuda_set_camera_resolution(image_width_, image_height_);
        // cuda初始化
        //  cuda_malloc_memory();
        cuda_malloc_basic_memory();
        cuda_malloc_hdr_memory();
        cuda_malloc_repetition_memory();

        if (!loadCalibData())
        {
            LOG(INFO)<<"Load Calib Error!"; 
            return DF_FAILED;
        }
    }
    else
    {

        return DF_FAILED;
    }


    int formate = 0;
    bool ok= camera_->getPixelFormat(formate);

    LOG(INFO)<<"getPixelFormat: "<<ok;

    return ret;
 
} 


int Scan3D::reopenCamera()
{
    LOG(ERROR)<<"reopen Camera:";
    if(!camera_->closeCamera())
    {
        LOG(ERROR)<<"close Camera Error!";
    }

    if(camera_->openCamera())
    {
        camera_->setExposure(camera_exposure_);
    }
    else
    {
        return DF_ERROR_2D_CAMERA;
    }
   
    LOG(ERROR)<<"reopen Camera finished!";
    return DF_SUCCESS;    
}

int Scan3D::initCache()
{
    std::memset(buff_brightness_, 0, sizeof(char) * image_width_ * image_height_);
    std::memset(buff_depth_, 0, sizeof(float) * image_width_ * image_height_);
    std::memset(buff_pointcloud_, 0, sizeof(float) * image_width_ * image_height_ * 3);

    return DF_SUCCESS;
}

bool Scan3D::cameraIsValid()
{
    return camera_opened_flag_;
}

bool Scan3D::triggerLineIsValid()
{
    return true;
}

bool Scan3D::setParamHdr(int num,std::vector<int> led_list,std::vector<int> exposure_list)
{
    if(led_list.size() != exposure_list.size() || exposure_list.size() != 6)
    {
        return false;
    }

    hdr_num_ = num;

    led_current_list_ = led_list;
    camera_exposure_list_ = exposure_list;

    lc3010_->set_hdr_list(hdr_num_, camera_exposure_list_, led_current_list_);

    return true;
}

bool Scan3D::setParamExposure(float exposure)
{ 
 
    if(exposure > max_camera_exposure_ || exposure < min_camera_exposure_)
    {
        return false;
    }

    if (!camera_->setExposure(exposure))
    {
        return false;
    }

    lc3010_->set_camera_exposure(exposure);
  
    camera_exposure_ = exposure;

    return true;
}

bool Scan3D::setParamGain(float gain)
{
    if (!camera_->setGain(gain))
    {
         return false;
    }

    camera_gain_ = gain;
    
    return true;
}


bool Scan3D::setParamLedCurrent(int current)
{

    if(DF_SUCCESS!= lc3010_->SetLedCurrent(current,current,current))
    {
        LOG(ERROR)<<"Set Led Current failed";
        return false;
    }

    led_current_ = current;
    LOG(INFO)<<"led_current: "<<led_current_;

    return true;
}


bool Scan3D::setParamConfidence(float confidence)
{
    return cuda_set_param_confidence(confidence);  
}

bool Scan3D::setProjectorVersion(int version)
{
    switch (version)
    {
    case DF_PROJECTOR_3010:
    {
        cuda_set_projector_version(DF_PROJECTOR_3010);
        max_camera_exposure_ = 100000;
        min_camera_exposure_ = 1700;
        return true;
    }
    break;

    case DF_PROJECTOR_4710:
    {

        cuda_set_projector_version(DF_PROJECTOR_4710);
        max_camera_exposure_ = 28000; 
        min_camera_exposure_ = 1700;
        return true;
    }
    break;

    default:
        break;
    }

    return false;
}


void Scan3D::getProjectorVersion(int &version)
{
    version = projector_version_;
}

bool Scan3D::setParamGenerateBrightness(int model, int exposure)
{
    if (model == 1 || model == 2 || model == 3|| model == 4)
    {
        generate_brightness_model_ = model;
        generate_brightness_exposure_ = exposure;

        return true;
    }

    return false;
}

void Scan3D::setParamSystemConfig(SystemConfigDataStruct param)
{
    system_config_settings_machine_ = param;
    cuda_set_param_system_config(param);
}

void Scan3D::setParamFisherConfidence(float confidence)
{
    fisher_confidence_val_ = confidence; 
}
/******************************************************************************************************************************************/


bool Scan3D::captureTextureImage(int model,float exposure,unsigned char* buff)
{
    // unsigned short* p_data= new unsigned short[image_width_*image_height_];
    cv::Mat short_data(image_height_, image_width_, CV_16UC1, cv::Scalar(0));

    int bits = 0;
    if(DF_SUCCESS != getPixelFormat(bits))
    {
        bits = 8;
    }


    switch (model)
    {
    case 1:
    {
 
        setParamExposure(exposure);
        camera_->switchToExternalTriggerMode();
        camera_->streamOn();
        LOG(INFO) << "Stream On";
        lc3010_->start_pattern_sequence();

        if(8 == bits)
        {
            if (!camera_->grap(buff))
            {
                LOG(INFO) << "grap brightness failed!";
            }
            else
            {
                LOG(INFO) << "grap brightness!";
            }
        }
        else
        {
            if (!camera_->grap((ushort*)short_data.data))
            {
                LOG(INFO) << "grap brightness failed!";
            }
            else
            {
                LOG(INFO) << "grap brightness!";
            }

            for (int r = 0; r < image_height_; r++)
            {

                for (int c = 0; c < image_width_; c++)
                {
                    buff[r * image_width_ + c] = short_data.at<ushort>(r, c) / 16 + 0.5;
                }
            }
        }


        camera_->streamOff();
        LOG(INFO) << "Stream Off";
    }
    break;
    case 2:
    {

        // 发光，自定义曝光时间
        lc3010_->enable_solid_field();

        switch (system_config_settings_machine_.Instance().firwmare_param_.generate_brightness_exposure_model)
        {
        case  GENERATE_BRIGHTNESS_MODEL_SINGLE_:
        {
            camera_->switchToInternalTriggerMode();
            camera_->setExposure(exposure);
            camera_->setGain(system_config_settings_machine_.Instance().firwmare_param_.brightness_gain);
            camera_->streamOn();
            LOG(INFO) << "Stream On";

            camera_->trigger_software();
            if(8 == bits)
            {
                if (!camera_->grap(buff))
                {
                    LOG(INFO) << "grap brightness failed!";
                }
                else
                {
                    LOG(INFO) << "grap brightness!";
                }
            }
            else
            {
                if (!camera_->grap((ushort*)short_data.data))
                {
                    LOG(INFO) << "grap brightness failed!";
                }
                else
                {
                    LOG(INFO) << "grap brightness!";
                }

                for(int r = 0;r< image_height_;r++)
                {
                    
                    for(int c = 0;c < image_width_;c++)
                    {
                        buff[r*image_width_+ c] = short_data.at<ushort>(r,c)/16 + 0.5;

                    }

                }

            }
            camera_->streamOff();
            LOG(INFO) << "Stream Off";
        }
        break;
        case GENERATE_BRIGHTNESS_MODEL_HDR_:
        {

            camera_->streamOn();
            LOG(INFO) << "Stream On";

            camera_->switchToInternalTriggerMode();
            camera_->setGain(system_config_settings_machine_.Instance().firwmare_param_.brightness_gain);

            int capture_num = system_config_settings_machine_.Instance().firwmare_param_.brightness_hdr_exposure_num;

            std::vector<cv::Mat> img_list;

            for (int i = 0; i < capture_num; i++)
            {
                cv::Mat img(image_height_, image_width_, CV_8UC1, cv::Scalar(0));

                int exposure_val = system_config_settings_machine_.Instance().firwmare_param_.brightness_hdr_exposure_param_list[i];
                LOG(INFO) << "set brightness exposure: " << exposure_val;
                camera_->setExposure(exposure_val);

                camera_->trigger_software();

                if (8 == bits)
                {
                    if (!camera_->grap(img.data))
                    {
                        LOG(INFO) << "grap brightness failed!";
                    }
                    else
                    {
                        LOG(INFO) << "grap brightness!";
                    }
                    img_list.push_back(img.clone());
                }
                else
                {
                     cv::Mat img_16(image_height_, image_width_, CV_16UC1, cv::Scalar(0));
                    if (!camera_->grap(( unsigned short*)img_16.data))
                    {
                        LOG(INFO) << "grap brightness failed!";
                    }
                    else
                    {
                        LOG(INFO) << "grap brightness!";
                    }

                    img = img_16/16;
                    img_list.push_back(img.clone());
                }

                //  LOG(INFO) << "pixels: " << cv::sum(img);

                // std::string path = "b_"+std::to_string(exposure_val) + ".bmp";
                // cv::imwrite(path,img);
            }

            camera_->streamOff();
            LOG(INFO) << "Stream Off";
            /********************************************************************************************/
            LOG(INFO) << "process: " << img_list.size();
            cv::Mat exposureFusion;
            cv::Ptr<cv::MergeMertens> mergeMertens = cv::createMergeMertens();
            mergeMertens->process(img_list, exposureFusion);

            for (int r = 0; r < image_height_; r++)
            {
                float *ptr_fusion = exposureFusion.ptr<float>(r);

                for (int c = 0; c < image_width_; c++)
                {
                    if (ptr_fusion[c] > 1)
                    {
                        buff[r * image_width_ + c] = 255;
                    }
                    else
                    {
                        buff[r * image_width_ + c] = 255 * ptr_fusion[c];
                    }
                }
            }
            LOG(INFO) << "merge finished!";
            /********************************************************************************************/
        }
        break;

        default:
            break;
        }

        lc3010_->disable_solid_field();
        camera_->switchToExternalTriggerMode();
        camera_->setExposure(camera_exposure_);
        camera_->setGain(camera_gain_);
    }
    break;
    case 3:
    {

        switch (system_config_settings_machine_.Instance().firwmare_param_.generate_brightness_exposure_model)
        {
        case GENERATE_BRIGHTNESS_MODEL_SINGLE_:
        {
            camera_->switchToInternalTriggerMode();
            if (!camera_->setExposure(exposure))
            {
                LOG(INFO) << "setExposure Failed!";
            }
            else
            {
                LOG(INFO) << "set Exposure: " << exposure;
            }

            camera_->setGain(system_config_settings_machine_.Instance().firwmare_param_.brightness_gain);

            camera_->streamOn();
            LOG(INFO) << "Stream On";

            camera_->trigger_software();

            if(8 == bits)
            {
                if (!camera_->grap(buff))
                {
                    LOG(INFO) << "grap brightness failed!";
                }
                else
                {
                    LOG(INFO) << "grap brightness!";
                }
            }
            else
            {
                if (!camera_->grap((ushort *)short_data.data))
                {
                    LOG(INFO) << "grap brightness failed!";
                }
                else
                {
                    LOG(INFO) << "grap brightness!";
                }

                for(int r = 0;r< image_height_;r++)
                {
                    
                    for(int c = 0;c < image_width_;c++)
                    {
                        buff[r*image_width_+ c] = short_data.at<ushort>(r,c)/16 + 0.5;

                    }

                }
            }

            camera_->streamOff();
            LOG(INFO) << "Stream Off";
        }
        break;

        case GENERATE_BRIGHTNESS_MODEL_HDR_:
        {

            camera_->streamOn();
            LOG(INFO) << "Stream On";

            camera_->switchToInternalTriggerMode();
            camera_->setGain(system_config_settings_machine_.Instance().firwmare_param_.brightness_gain);

            int capture_num = system_config_settings_machine_.Instance().firwmare_param_.brightness_hdr_exposure_num;

            std::vector<cv::Mat> img_list;

            for (int i = 0; i < capture_num; i++)
            {
                cv::Mat img(image_height_, image_width_, CV_8UC1, cv::Scalar(0));

                int exposure_val = system_config_settings_machine_.Instance().firwmare_param_.brightness_hdr_exposure_param_list[i];
                LOG(INFO) << "set brightness exposure: " << exposure_val;
                camera_->setExposure(exposure_val);

                camera_->trigger_software();

                if (8 == bits)
                {
                    if (!camera_->grap(img.data))
                    {
                        LOG(INFO) << "grap brightness failed!";
                    }
                    else
                    {
                        LOG(INFO) << "grap brightness!";
                    }
                    img_list.push_back(img.clone());
                }
                else
                {
                     cv::Mat img_16(image_height_, image_width_, CV_16UC1, cv::Scalar(0));
                    if (!camera_->grap(( unsigned short*)img_16.data))
                    {
                        LOG(INFO) << "grap brightness failed!";
                    }
                    else
                    {
                        LOG(INFO) << "grap brightness!";
                    }

                    img = img_16/16;
                    img_list.push_back(img.clone());
                }


                //  LOG(INFO) << "pixels: " << cv::sum(img);

                // std::string path = "b_"+std::to_string(exposure_val) + ".bmp";
                // cv::imwrite(path,img);
            }

            camera_->streamOff();
            LOG(INFO) << "Stream Off";
            /********************************************************************************************/
            LOG(INFO) << "process: " << img_list.size();
            cv::Mat exposureFusion;
            cv::Ptr<cv::MergeMertens> mergeMertens = cv::createMergeMertens();
            mergeMertens->process(img_list, exposureFusion);

            for (int r = 0; r < image_height_; r++)
            {
                float *ptr_fusion = exposureFusion.ptr<float>(r);

                for (int c = 0; c < image_width_; c++)
                {
                    if (ptr_fusion[c] > 1)
                    {
                        buff[r * image_width_ + c] = 255;
                    }
                    else
                    {
                        buff[r * image_width_ + c] = 255 * ptr_fusion[c];
                    }
                }
            }
            LOG(INFO) << "merge finished!";
            /********************************************************************************************/
        }
        break;

        default:
            break;
        }

        camera_->switchToExternalTriggerMode();
        camera_->setExposure(camera_exposure_);
        camera_->setGain(camera_gain_);
    }
    break;
    case 4:
    {

        // 发光，自定义曝光时间
        lc3010_->enable_solid_field();

        
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
 
        camera_->streamOn();
        LOG(INFO) << "Stream On";

        camera_->switchToInternalTriggerMode();
        camera_->setGain(system_config_settings_machine_.Instance().firwmare_param_.brightness_gain);

        int capture_num = system_config_settings_machine_.Instance().firwmare_param_.brightness_hdr_exposure_num;

        std::vector<cv::Mat> img_list;

        for (int i = 0; i < capture_num; i++)
        {
            cv::Mat img(image_height_, image_width_, CV_8UC1, cv::Scalar(0));

            int exposure_val = system_config_settings_machine_.Instance().firwmare_param_.brightness_hdr_exposure_param_list[i];
            LOG(INFO) << "set brightness exposure: " << exposure_val;
            camera_->setExposure(exposure_val);


            camera_->trigger_software();

                if (8 == bits)
                {
                    if (!camera_->grap(img.data))
                    {
                        LOG(INFO) << "grap brightness failed!";
                    }
                    else
                    {
                        LOG(INFO) << "grap brightness!";
                    }
                    img_list.push_back(img.clone());
                }
                else
                {
                     cv::Mat img_16(image_height_, image_width_, CV_16UC1, cv::Scalar(0));
                    if (!camera_->grap(( unsigned short*)img_16.data))
                    {
                        LOG(INFO) << "grap brightness failed!";
                    }
                    else
                    {
                        LOG(INFO) << "grap brightness!";
                    }

                    img = img_16/16;
                    img_list.push_back(img.clone());
                }


            //  LOG(INFO) << "pixels: " << cv::sum(img);

            // std::string path = "b_"+std::to_string(exposure_val) + ".bmp";
            // cv::imwrite(path,img);
        }

        camera_->streamOff();
        LOG(INFO) << "Stream Off";
        /********************************************************************************************/
        LOG(INFO) << "process: " << img_list.size();
        cv::Mat exposureFusion;
        cv::Ptr<cv::MergeMertens> mergeMertens = cv::createMergeMertens();
        mergeMertens->process(img_list, exposureFusion);

        for (int r = 0; r < image_height_; r++)
        {
            float *ptr_fusion = exposureFusion.ptr<float>(r);

            for (int c = 0; c < image_width_; c++)
            {
                if (ptr_fusion[c] > 1)
                {
                    buff[r * image_width_ + c] = 255;
                }
                else
                {
                    buff[r * image_width_ + c] = 255 * ptr_fusion[c];
                }
            }
        }
        LOG(INFO) << "merge finished!";
        /********************************************************************************************/

        lc3010_->disable_solid_field();
        camera_->switchToExternalTriggerMode();
        camera_->setExposure(camera_exposure_);
        camera_->setGain(camera_gain_);
    }
    break;

    default:
        break;
    }

    std::cout << "_____________end_______________" << std::endl;
    return true;
}

/*******************************************************************************************************************************************/

bool Scan3D::captureRaw01(unsigned char* buff)
{

    int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(8 != bits)
        {
            if(setPixelFormat(8))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }

    lc3010_->pattern_mode01();
    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        return false;
    }

    lc3010_->start_pattern_sequence();

    int img_size = image_width_*image_height_;

    unsigned char *img_ptr= new unsigned char[image_width_*image_height_];

    for (int i = 0; i < 24; i++)
    {
        LOG(INFO)<<"grap "<<i<<" image:";
        if (!camera_->grap(img_ptr))
        {
            camera_->streamOff();
            return false;
        }
 
        memcpy(buff+img_size*i, img_ptr, img_size);
  
    }

    camera_->streamOff();

        
    if (1 != generate_brightness_model_)
    { 
        captureTextureImage(generate_brightness_model_, generate_brightness_exposure_,img_ptr);
        memcpy(buff+img_size*23, img_ptr, img_size);
    }
  
    delete []img_ptr;
}

bool Scan3D::captureRaw02(unsigned char* buff)
{
 
     int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(8 != bits)
        {
            if(setPixelFormat(8))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }

    lc3010_->pattern_mode02();
    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        return false;
    }

    lc3010_->start_pattern_sequence();

    int img_size = image_width_*image_height_;

    unsigned char *img_ptr= new unsigned char[image_width_*image_height_];

    for (int i = 0; i < 37; i++)
    {
        LOG(INFO)<<"grap "<<i<<" image:";
        if (!camera_->grap(img_ptr))
        {
            camera_->streamOff();
            return false;
        }
        LOG(INFO) << "finished " << i;
 
        memcpy(buff+img_size*i, img_ptr, img_size);
  
    }

    camera_->streamOff();

    
    if (1 != generate_brightness_model_)
    { 
        captureTextureImage(generate_brightness_model_, generate_brightness_exposure_,img_ptr);
        memcpy(buff+img_size*36, img_ptr, img_size);
    }

    delete []img_ptr;
  
 
    return true;
}

bool Scan3D::captureRaw03(unsigned char* buff)
{

    int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(8 != bits)
        {
            if(setPixelFormat(8))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }

    lc3010_->pattern_mode03();
    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        return false;
    }

    lc3010_->start_pattern_sequence();

    int img_size = image_width_*image_height_;

    unsigned char *img_ptr= new unsigned char[image_width_*image_height_];

    for (int i = 0; i < 31; i++)
    {
        LOG(INFO)<<"grap "<<i<<" image:";
        if (!camera_->grap(img_ptr))
        {
            camera_->streamOff();
            return false;
        }
 
        memcpy(buff+img_size*i, img_ptr, img_size);
  
    }

    camera_->streamOff();
    if (1 != generate_brightness_model_)
    {
        captureTextureImage(generate_brightness_model_, generate_brightness_exposure_, img_ptr);
        memcpy(buff + img_size * 30, img_ptr, img_size);
    }

    delete []img_ptr;
 
    return true;
}

bool Scan3D::captureRaw04(unsigned char* buff)
{

    int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(8 != bits)
        {
            if(setPixelFormat(8))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }

    lc3010_->pattern_mode04();
    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        return false;
    }

    lc3010_->start_pattern_sequence();

    int img_size = image_width_*image_height_;

    unsigned char *img_ptr= new unsigned char[image_width_*image_height_];

    for (int i = 0; i < 19; i++)
    {
        LOG(INFO)<<"grap "<<i<<" image:";
        if (!camera_->grap(img_ptr))
        {
            camera_->streamOff();
            return false;
        }
 
        memcpy(buff+img_size*i, img_ptr, img_size);
  
    }

    camera_->streamOff();

    if (1 != generate_brightness_model_)
    {
        captureTextureImage(generate_brightness_model_, generate_brightness_exposure_, img_ptr);
        memcpy(buff + img_size * 18, img_ptr, img_size);
    }

    delete[] img_ptr;

    return true;
}

int Scan3D::captureRaw05(unsigned char *buff)
{

    int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(8 != bits)
        {
            if(setPixelFormat(8))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }

    int patterns_num = 16;

    lc3010_->pattern_mode05();
    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        return false;
    }

    lc3010_->start_pattern_sequence();

    int img_size = image_width_*image_height_;

    unsigned char *img_ptr= new unsigned char[image_width_*image_height_];

    for (int i = 0; i < patterns_num; i++)
    {
        LOG(INFO)<<"grap "<<i<<" image:";
        if (!camera_->grap(img_ptr))
        {
            camera_->streamOff();
            return false;
        }
 
        memcpy(buff+img_size*i, img_ptr, img_size);
  
    }

    camera_->streamOff();
 
    delete[] img_ptr;

    return true;
}

int Scan3D::captureRaw06(unsigned char *buff)
{

    int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(8 != bits)
        {
            if(setPixelFormat(8))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }

   int patterns_num = 16;

    lc3010_->pattern_mode06();
    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        return false;
    }

    lc3010_->start_pattern_sequence();

    int img_size = image_width_*image_height_;

    unsigned char *img_ptr= new unsigned char[image_width_*image_height_];

    for (int i = 0; i < patterns_num; i++)
    {
        LOG(INFO)<<"grap "<<i<<" image:";
        if (!camera_->grap(img_ptr))
        {
            camera_->streamOff();
            return false;
        }
 
        memcpy(buff+img_size*i, img_ptr, img_size);
  
    }

    camera_->streamOff();
 
    delete[] img_ptr;

    return true;
}

int Scan3D::captureRaw08(unsigned char *buff)
{
    int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(8 != bits)
        {
            if(setPixelFormat(8))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }


    lc3010_->pattern_mode08();
    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        return DF_ERROR_CAMERA_STREAM;
    }

    lc3010_->start_pattern_sequence();

    int img_size = image_width_*image_height_;

    unsigned char *img_ptr= new unsigned char[image_width_*image_height_];

    for (int i = 0; i < 24; i++)
    {
        LOG(INFO)<<"grap "<<i<<" image:";
        if (!camera_->grap(img_ptr))
        {
            camera_->streamOff();
            return DF_ERROR_CAMERA_GRAP;
        }
 
        memcpy(buff+img_size*i, img_ptr, img_size);
  
    }

    camera_->streamOff();

    if (1 != generate_brightness_model_)
    {
        captureTextureImage(generate_brightness_model_, generate_brightness_exposure_, img_ptr);
        memcpy(buff + img_size * 0, img_ptr, img_size);
    }

    delete[] img_ptr;
 

    return DF_SUCCESS;
}

bool Scan3D::captureRaw04Repetition01(int repetition_count,unsigned char* buff)
{
    
    // lc3010_->pattern_mode04_repetition(repetition_count);
    // if (!camera_->streamOn())
    // {
    //     LOG(INFO) << "Stream On Error";
    //     return false;
    // }

    // lc3010_->start_pattern_sequence();

    // int img_size = image_width_*image_height_;

    // unsigned char *img_ptr= new unsigned char[image_width_*image_height_]; 

    // int capture_num= 19 + 6*(repetition_count-1);

    // for (int i = 0; i < capture_num; i++)
    // {
    //     LOG(INFO)<<"grap "<<i<<" image:";
    //     if (!camera_->grap(img_ptr))
    //     {
    //         camera_->streamOff();
    //         return false;
    //     }
 
    //     memcpy(buff+img_size*i, img_ptr, img_size);
  
    // }

    // delete []img_ptr;
    // camera_->streamOff();
   
    return true;
}


bool Scan3D::capturePhase02Repetition02(int repetition_count,float* phase_x,float* phase_y,unsigned char* brightness)
{
    cuda_clear_repetition_02_patterns();

    int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(8 != bits)
        {
            if(setPixelFormat(8))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }

    unsigned char *img_ptr= new unsigned char[image_width_*image_height_];

    for (int r = 0; r < repetition_count; r++)
    {
        int n = 0;
  
            LOG(INFO) << "pattern_mode02";
            lc3010_->pattern_mode02();
            
            if(!camera_->streamOn())
            {
                LOG(INFO) << "stream on failed!";
                return false;
            } 
            lc3010_->start_pattern_sequence();
            LOG(INFO) << "start_pattern_sequence";

            for (int i = 0; i < 37; i++)
            {
                LOG(INFO) << "receiving " << i << "th image";
                bool status = camera_->grap(img_ptr);
                LOG(INFO) << "status=" << status;

                if (status)
                {
                    if (i != 36 || 1 == generate_brightness_model_)
                    {
                        cuda_copy_pattern_to_memory(img_ptr, i);
                        cuda_merge_repetition_02_patterns(i);
                    }
                }
                else
                {
                    LOG(INFO) << "grad failed!";
                    camera_->streamOff(); 
                    delete []img_ptr;
                    return false;
                }
 
     
            }

            camera_->streamOff(); 

            if (1 != generate_brightness_model_)
            {

                captureTextureImage(generate_brightness_model_, generate_brightness_exposure_, img_ptr);

                cuda_copy_pattern_to_memory(img_ptr, 36);
                cuda_merge_repetition_02_patterns(36);
            }

            /***********************************************************************************************/ 
    }


    delete []img_ptr;

    cuda_compute_merge_repetition_02_phase(repetition_count, 2);
    LOG(INFO) << "parallel_cuda_compute_mergerepetition_02_phase";
    cuda_unwrap_phase_shift(1);
    cuda_unwrap_phase_shift(2);
    cuda_unwrap_phase_shift(3);
    cuda_unwrap_phase_shift(5);
    cuda_unwrap_phase_shift(6);
    cuda_unwrap_phase_shift(7);
    cuda_normalize_phase(0);
    cuda_normalize_phase(2);

    cuda_copy_phase_from_cuda_memory(phase_x, phase_y);
    cuda_copy_brightness_from_memory(brightness); 
 
    return true;
}

// bool Scan3D::captureFrame04()
// {
 
//     lc3010_->pattern_mode04();
//     LOG(INFO) << "Stream On:";
//     if (!camera_->streamOn())
//     {
//         LOG(INFO) << "Stream On Error";
//         return false;
//     }

//     lc3010_->start_pattern_sequence();

//     unsigned char *img_ptr= new unsigned char[image_width_*image_height_];

//     for (int i = 0; i < 19; i++)
//     {
//         LOG(INFO)<<"grap "<<i<<" image:";
//         if (!camera_->grap(img_ptr))
//         {
            
//             delete[] img_ptr; 
//             camera_->streamOff();
//             return false;
//         }
//         LOG(INFO)<<"finished!";

//         if(18 == i)
//         {
//             cuda_copy_brightness_to_memory(img_ptr);
//         }

//         cuda_copy_pattern_to_memory(img_ptr, i);

//         // copy to gpu
//         switch (i)
//         {
//         case 4:
//         {
//         LOG(INFO)<<"cuda_compute_phase_shift:";
//             cuda_compute_phase_shift(0); 
//         }
//         break;
//         case 8:
//         {
//             cuda_compute_phase_shift(1);
//         }
//         break;
//         case 10:
//         {
//             cuda_unwrap_phase_shift(1);
//         }
//         break;
//         case 12:
//         {
//             cuda_compute_phase_shift(2);
//         }
//         break;
//         case 15:
//         {
//             cuda_unwrap_phase_shift(2);
//         }
//         break;
//         case 18:
//         {


//             cuda_compute_phase_shift(3);
//             cuda_unwrap_phase_shift(3);
//             cuda_normalize_phase(0);

//             cuda_generate_pointcloud_base_table();
//             //  cudaDeviceSynchronize();
//             LOG(INFO) << "cuda_generate_pointcloud_base_table";
//         }

//         default:
//             break;
//         }
//     }

//     delete[] img_ptr;

//     camera_->streamOff();
//     LOG(INFO) << "Stream Off";

 
//     cuda_copy_depth_from_memory(buff_depth_);
//     cuda_copy_pointcloud_from_memory(buff_pointcloud_);

//     if (1 == generate_brightness_model_)
//     { 
//         cuda_copy_brightness_from_memory(buff_brightness_);
//     }
//     else
//     {

//         captureTextureImage(generate_brightness_model_, generate_brightness_exposure_,buff_brightness_);
//     }

 
//     return true;
// }


int Scan3D::captureFrame04BaseConfidence()
{
    if (hdr_is_shooting_.try_lock()) { lc3010_->disable_hdr_mode(); hdr_is_shooting_.unlock(); }
    LOG(INFO) << "cuda_clear_reconstruct_cache:";
    cuda_clear_reconstruct_cache();
    initCache();

    int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(8 != bits)
        {
            if(setPixelFormat(8))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }


    int ret = DF_SUCCESS;
    //lc3010_->start_pattern_sequence();
    //{
    //    return DF_SUCCESS;
    //}
    ret = lc3010_->pattern_mode04();
    if(DF_SUCCESS != ret)
    {
        return ret;
    }
 

    LOG(INFO) << "Stream On:";
    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        return DF_ERROR_CAMERA_STREAM;
    }

    lc3010_->start_pattern_sequence();

    unsigned char *img_ptr= new unsigned char[image_width_*image_height_];
    
    for (int i = 0; i < 19; i++)
    {
        std::cout << "1" << std::endl;
        LOG(INFO)<<"grap "<<i<<" image:";
        if (!camera_->grap(img_ptr))
        {
            
            delete[] img_ptr; 
            camera_->streamOff(); 
            if(i == 0)
            {
                return DF_ERROR_LOST_TRIGGER;
            }

            return DF_ERROR_CAMERA_GRAP;
        }
        //cv::Mat brightness_mat(image_height_, image_width_, CV_8U, img_ptr);
        //std::string filename = std::to_string(i) + ".bmp";
        //cv::imwrite(filename, brightness_mat);
        LOG(INFO)<<"finished!";

        if(18 == i)
        {
            cuda_copy_brightness_to_memory(img_ptr);
        }

        cuda_copy_pattern_to_memory(img_ptr, i);

        // copy to gpu
        switch (i)
        {
        case 4:
        {
        LOG(INFO)<<"cuda_compute_phase_shift:";
            cuda_compute_phase_shift(0); 
        }
        break;
        case 8:
        {
            cuda_compute_phase_shift(1);
        }
        break;
        case 10:
        {
            cuda_unwrap_phase_shift_base_fisher_confidence(1);
        }
        break;
        case 12:
        {
            cuda_compute_phase_shift(2);
        }
        break;
        case 15:
        {
            cuda_unwrap_phase_shift_base_fisher_confidence(2);
        }
        break;
        case 18:
        {


            cuda_compute_phase_shift(3);
            if (1 == system_config_settings_machine_.Instance().firwmare_param_.use_gray_rectify)
            {
                cv::Mat convolution_kernal = cv::getGaussianKernel(system_config_settings_machine_.Instance().firwmare_param_.gray_rectify_r, system_config_settings_machine_.Instance().firwmare_param_.gray_rectify_sigma * 0.02, CV_32F);
	            convolution_kernal = convolution_kernal * convolution_kernal.t();
                cuda_copy_convolution_kernal_to_memory((float*)convolution_kernal.data, system_config_settings_machine_.Instance().firwmare_param_.gray_rectify_r);
                cuda_rectify_six_step_pattern_phase(0, system_config_settings_machine_.Instance().firwmare_param_.gray_rectify_r);
            }
            cuda_unwrap_phase_shift_base_fisher_confidence(3);
            LOG(INFO) << "fisher_confidence_val： " << fisher_confidence_val_;
            //if (-50 < fisher_confidence_val_)
            //{
            //    fisher_filter(fisher_confidence_val_);
            //}
            cuda_normalize_phase(0);

            cuda_generate_pointcloud_base_table();
            LOG(INFO) << "cuda_generate_pointcloud_base_table";
        }

        default:
            break;
        }
    }

    delete[] img_ptr;

    camera_->streamOff();
    LOG(INFO) << "Stream Off";
    
    cuda_copy_depth_from_memory(buff_depth_);
    cuda_copy_pointcloud_from_memory(buff_pointcloud_);

    if (1 == generate_brightness_model_)
    { 
        cuda_copy_brightness_from_memory(buff_brightness_);
    }

    return DF_SUCCESS;
}


int Scan3D::captureFrame06Repetition(int repetition_count)
{
        if (hdr_is_shooting_.try_lock()) { lc3010_->disable_hdr_mode(); hdr_is_shooting_.unlock(); }
    LOG(INFO) << "cuda_clear_reconstruct_cache:";
    cuda_clear_reconstruct_cache();
    initCache();

    int frame_status = DF_SUCCESS;
    cuda_clear_repetition_02_patterns();



    if (patterns_sets_num_ < 9)
    {
        return DF_ERROR_LOST_PATTERN_SETS;
    }

    int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(8 != bits)
        {
            if(setPixelFormat(8))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }


    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        frame_status = DF_ERROR_CAMERA_STREAM;
        return DF_ERROR_CAMERA_STREAM;
    }

    unsigned char *img_ptr= new unsigned char[image_width_*image_height_];

    for(int r= 0;r< repetition_count;r++)
    {
        int n = 0;
  
        LOG(INFO) << "pattern_mode06";
        int ret = lc3010_->pattern_mode06();
        if(DF_SUCCESS != ret)
        {
            frame_status = ret;
            camera_->streamOff();
            return ret;
        }

        lc3010_->start_pattern_sequence();
        LOG(INFO) << "start_pattern_sequence";

        for (int g_i = 0; g_i < 16; g_i++)
        {
            LOG(INFO) << "receiving " << g_i << "th image";
            bool status = camera_->grap(img_ptr);
            LOG(INFO) << "status=" << status;

            if (status)
            {

            if (0 == g_i)
            {
                cuda_copy_brightness_to_memory(img_ptr);
            }
                cuda_copy_minsw8_pattern_to_memory(img_ptr, g_i);
                // cuda_copy_pattern_to_memory(img_ptr, i);
                cuda_merge_repetition_02_patterns(g_i);
            }
            else
            {
                LOG(INFO) << "grad failed!";
                camera_->streamOff();
                delete[] img_ptr;

                if (g_i == 0)
                {
                    return DF_ERROR_LOST_TRIGGER;
                }

                frame_status = DF_ERROR_CAMERA_GRAP;
                return DF_ERROR_CAMERA_GRAP;
            }
        }

        /*********************************************************************************************/

        /***********************************************************************************************/
    }

    camera_->streamOff();
    LOG(INFO) << "GXStreamOff";

    delete[] img_ptr;

    cuda_handle_repetition_model06(repetition_count); 
    if (fisher_confidence_val_ > -50)
    {
        phase_monotonicity_filter(fisher_confidence_val_ / 2 + 25);
    }
    cuda_normalize_phase(0); 
    LOG(INFO) << "parallel_cuda_unwrap_phase";
    cuda_generate_pointcloud_base_table();
    // depth_filter(system_config_settings_machine_.Instance().firwmare_param_.radius_filter_r / 1000.);
    LOG(INFO) << "generate_pointcloud_base_table";


    cuda_copy_depth_from_memory(buff_depth_);
    cuda_copy_pointcloud_from_memory(buff_pointcloud_);

    if (1 == generate_brightness_model_)
    { 
        cuda_copy_brightness_from_memory(buff_brightness_);
    }
    // else
    // {

    //     captureTextureImage(generate_brightness_model_, generate_brightness_exposure_,buff_brightness_);
    // }

    return frame_status;
}

int Scan3D::captureFrame06Repetition_cpu(int repetition_count)
{
    if (hdr_is_shooting_.try_lock()) { lc3010_->disable_hdr_mode(); hdr_is_shooting_.unlock(); }
    LOG(INFO) << "cuda_clear_reconstruct_cache:";
    cuda_clear_reconstruct_cache();
    initCache();

    int frame_status = DF_SUCCESS;
    cuda_clear_repetition_02_patterns();



    if (patterns_sets_num_ < 9)
    {
        return DF_ERROR_LOST_PATTERN_SETS;
    }

    int bits = 0;
    if (DF_SUCCESS == getPixelFormat(bits))
    {
        if (8 != bits)
        {
            if (setPixelFormat(8))
            {
                LOG(INFO) << "set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }


    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        frame_status = DF_ERROR_CAMERA_STREAM;
        return DF_ERROR_CAMERA_STREAM;
    }

    int nr = image_height_;
    int nc = image_width_;

    cv::Mat merge_brightness_map(nr, nc, CV_16U, cv::Scalar(0));

    std::vector<cv::Mat>repetition_02_merge_patterns_list;
    repetition_02_merge_patterns_list.resize(16);
    for (cv::Mat& mat : repetition_02_merge_patterns_list) {
        mat = cv::Mat(nr, nc, CV_16U, cv::Scalar(0)); // 使用 cv::Scalar(0) 将所有像素初始化为 0  
    }
   
    unsigned char* img_ptr = new unsigned char[image_width_ * image_height_];
 

    for (int r = 0; r < repetition_count; r++)
    {
        int n = 0;

        LOG(INFO) << "pattern_mode06";
        int ret = lc3010_->pattern_mode06();
        if (DF_SUCCESS != ret)
        {
            frame_status = ret;
            camera_->streamOff();
            return ret;
        }

        lc3010_->start_pattern_sequence();
        LOG(INFO) << "start_pattern_sequence";
  
        std::vector<cv::Mat> patterns_;

        for (int g_i = 0; g_i < 16; g_i++)
        {
            LOG(INFO) << "receiving " << g_i << "th image";
            bool status = camera_->grap(img_ptr);
            LOG(INFO) << "status=" << status;

            if (status)
            {


                cv::Mat img(image_height_, image_width_, CV_8UC1, img_ptr);
                if (g_i < 2 || g_i > 7) {
                    cv::GaussianBlur(img, img, cv::Size(5, 5), 1, 1);
                    std::cout << g_i << "gaussianblur" << std::endl;
                }
                patterns_.push_back(img.clone());
           
                cuda_merge_repetition_02_patterns_cpu(nr, nc, g_i, patterns_, merge_brightness_map, repetition_02_merge_patterns_list);
            }
            else
            {
                LOG(INFO) << "grad failed!";
                camera_->streamOff();
                delete[] img_ptr;

                if (g_i == 0)
                {
                    return DF_ERROR_LOST_TRIGGER;
                }

                frame_status = DF_ERROR_CAMERA_GRAP;
                return DF_ERROR_CAMERA_GRAP;
            }
        }
        LOG(INFO) << "GXStreamOff";
        /*********************************************************************************************/

        /***********************************************************************************************/
    }

    camera_->streamOff();
    LOG(INFO) << "GXStreamOff";

    delete[] img_ptr;


    std::vector<cv::Mat> patterns_;
    cv::Mat threshold_map(nr, nc, CV_16U, cv::Scalar(128));
    cv::Mat mask;
    cv::Mat wrap;
    cv::Mat sw_k_map(nr, nc, CV_8U, cv::Scalar(0));
    cv::Mat minsw_map(nr, nc, CV_32F, cv::Scalar(0));
    cv::Mat k2_map;
    cv::Mat unwrap;
    cv::Mat deep_map_table;
    cv::Mat xyz_mat;


    cuda_handle_repetition_model06_cpu(repetition_count,  nr,  nc,repetition_02_merge_patterns_list,
        threshold_map,  mask,  wrap, sw_k_map,  minsw_map,  k2_map,  unwrap);
    
    

    //if (fisher_confidence_val_ > -50)
    //{
    //    phase_monotonicity_filter(fisher_confidence_val_ / 2 + 25);
    //}

    float ver_period = 128.0;
    unwrap /= ver_period;

    LOG(INFO) << "parallel_cuda_unwrap_phase";


    LookupTableFunction lookup_table_machine_1;
    cv::Mat xL_rotate_x1;
    cv::Mat xL_rotate_y1;
    cv::Mat rectify_R11;
    cv::Mat pattern_mapping1;
    cv::Mat pattern_minimapping1(128, 128, CV_64FC1, cv::Scalar(0));

    bool read_map_ok = lookup_table_machine_1.readTableFloat("./", xL_rotate_x1, xL_rotate_y1, rectify_R11, pattern_mapping1, pattern_minimapping1, image_width_, image_height_);
    float b = sqrt(pow(calib_param_.translation_matrix[0], 2) + pow(calib_param_.translation_matrix[1], 2) + pow(calib_param_.translation_matrix[2], 2));

    lookup_table_machine_1.generate_pointcloud_depth_repetition(mask, unwrap, deep_map_table, xyz_mat, xL_rotate_x1, xL_rotate_y1, rectify_R11, pattern_mapping1, b);

    LOG(INFO) << "generate_pointcloud_base_table";
    
    for (int r = 0; r < nr; r++)
    {
        double* ptr_w = deep_map_table.ptr<double>(r);
        ushort* bright = merge_brightness_map.ptr<ushort>(r);
        for (int c = 0; c < nc; c++)
        {
            buff_depth_[r * deep_map_table.cols + c] = static_cast<double>(ptr_w[c]);
            buff_brightness_[r * merge_brightness_map.cols + c] = 0.5+(static_cast<ushort>(bright[c])/ repetition_count);
        }
    }

    //cuda_copy_depth_from_memory(buff_depth_);
    //cuda_copy_pointcloud_from_memory(buff_pointcloud_);

  /*  if (1 == generate_brightness_model_)
    {
        cuda_copy_brightness_from_memory(buff_brightness_);
    }*/
    // else
    // {

    //     captureTextureImage(generate_brightness_model_, generate_brightness_exposure_,buff_brightness_);
    // }

    return frame_status;
}




	int  Scan3D::bayerToRgb(unsigned short* src, unsigned short* dst)
	{

		int height = image_height_;
		int width = image_width_;

		//按3×3的邻域处理，边缘需填充至少1个像素的宽度
		int nBorder = 1;
		unsigned short* bayer = (unsigned short*)malloc(sizeof(unsigned short) * (width + 2 * nBorder) * (height + 2 * nBorder));
		memset(bayer, 0, sizeof(unsigned short) * (width + 2 * nBorder) * (height + 2 * nBorder));

		for (int r = 0; r < height; r++)
		{
			for (int c = 0; c < width; c++)
			{
				bayer[(r + nBorder) * (width + 2 * nBorder) + (c + nBorder)] = src[r * width + c];
			}
		}

		for (int b = 0; b < nBorder; b++)
		{
			for (int r = 0; r < height; r++)
			{
				bayer[(r + nBorder) * (width + 2 * nBorder) + b] = src[r * width];
				bayer[(r + nBorder) * (width + 2 * nBorder) + b + 2 * nBorder + width - 1] = src[r * width + width - 1];
			}
		}

		for (int b = 0; b < nBorder; b++)
		{
			for (int c = 0; c < width + 2 * nBorder; c++)
			{
				bayer[b * (width + 2 * nBorder) + c] = bayer[nBorder * (width + 2 * nBorder) + c];
				bayer[(nBorder + height + b) * (width + 2 * nBorder) + c] = bayer[(nBorder + height - 1) * (width + 2 * nBorder) + c];
			}
		}

		unsigned short* p_rgb = (unsigned short*)malloc(sizeof(unsigned short) * 3 * (width + 2 * nBorder) * (height + 2 * nBorder));
		memset(p_rgb, 0, sizeof(unsigned short) * 3 * (width + 2 * nBorder) * (height + 2 * nBorder));


		unsigned short* pBayer = bayer;
		unsigned short* pRGB = p_rgb;
		int nW = width + 2 * nBorder;
		int nH = height + 2 * nBorder;


		for (int i = nBorder; i < nH - nBorder; i++)
		{
			for (int j = nBorder; j < nW - nBorder; j++)
			{
				//3×3邻域像素定义
				/*
				 * |M00 M01 M02|
				 * |M10 M11 M12|
				 * |M20 M21 M22|
				*/
				int nM00 = (i - 1) * nW + (j - 1); int nM01 = (i - 1) * nW + (j + 0);  int nM02 = (i - 1) * nW + (j + 1);
				int nM10 = (i - 0) * nW + (j - 1); int nM11 = (i - 0) * nW + (j + 0);  int nM12 = (i - 0) * nW + (j + 1);
				int nM20 = (i + 1) * nW + (j - 1); int nM21 = (i + 1) * nW + (j + 0);  int nM22 = (i + 1) * nW + (j + 1);

				if (i % 2 == 0)
				{
					if (j % 2 == 0)     //偶数行偶数列
					{
						pRGB[i * nW * 3 + j * 3 + 0] = pBayer[nM11];//b
						pRGB[i * nW * 3 + j * 3 + 2] = ((pBayer[nM00] + pBayer[nM02] + pBayer[nM20] + pBayer[nM22]) >> 2);//r
						pRGB[i * nW * 3 + j * 3 + 1] = ((pBayer[nM01] + pBayer[nM10] + pBayer[nM12] + pBayer[nM21]) >> 2);//g
					}
					else             //偶数行奇数列
					{
						pRGB[i * nW * 3 + j * 3 + 1] = pBayer[nM11];//g
						pRGB[i * nW * 3 + j * 3 + 2] = (pBayer[nM01] + pBayer[nM21]) >> 1;//r
						pRGB[i * nW * 3 + j * 3 + 0] = (pBayer[nM10] + pBayer[nM12]) >> 1;//b
					}
				}
				else
				{
					if (j % 2 == 0)     //奇数行偶数列
					{
						pRGB[i * nW * 3 + j * 3 + 1] = pBayer[nM11];//g
						pRGB[i * nW * 3 + j * 3 + 2] = (pBayer[nM10] + pBayer[nM12]) >> 1;//r
						pRGB[i * nW * 3 + j * 3 + 0] = (pBayer[nM01] + pBayer[nM21]) >> 1;//b
					}
					else             //奇数行奇数列
					{
						pRGB[i * nW * 3 + j * 3 + 2] = pBayer[nM11];//r
						pRGB[i * nW * 3 + j * 3 + 1] = (pBayer[nM01] + pBayer[nM21] + pBayer[nM10] + pBayer[nM12]) >> 2;//g
						pRGB[i * nW * 3 + j * 3 + 0] = (pBayer[nM00] + pBayer[nM02] + pBayer[nM20] + pBayer[nM22]) >> 2;//b
					}
				}
			}
		}


		for (int r = 0; r < height; r++)
		{
			for (int c = 0; c < width; c++)
			{
				dst[3 * (r * width + c) + 0] = pRGB[3 * ((r + nBorder) * (width + 2 * nBorder) + c + nBorder) + 0];
				dst[3 * (r * width + c) + 1] = pRGB[3 * ((r + nBorder) * (width + 2 * nBorder) + c + nBorder) + 1];
				dst[3 * (r * width + c) + 2] = pRGB[3 * ((r + nBorder) * (width + 2 * nBorder) + c + nBorder) + 2];
			}
		}


		free(bayer);
		free(p_rgb);

		return 0;
	}


int Scan3D::captureFrame06RepetitionColorMono12(int repetition_count)
{
    if (hdr_is_shooting_.try_lock()) { lc3010_->disable_hdr_mode(); hdr_is_shooting_.unlock(); }
    LOG(INFO) << "cuda_clear_reconstruct_cache:";
    cuda_clear_reconstruct_cache();
    initCache();

    int frame_status = DF_SUCCESS;
    cuda_clear_repetition_02_patterns();

    if (patterns_sets_num_ < 9)
    {
        return DF_ERROR_LOST_PATTERN_SETS;
    }

    int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(12 != bits)
        {
            if(setPixelFormat(12))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }
    // setPixelFormat(12);

    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        frame_status = DF_ERROR_CAMERA_STREAM;
        return DF_ERROR_CAMERA_STREAM;
    }

    // unsigned short *img_ptr = new unsigned short[image_width_ * image_height_];

    cv::Mat img(image_height_,image_width_,CV_16U,cv::Scalar(0));
    cv::Mat color_img(image_height_,image_width_,CV_16UC3,cv::Scalar(0,0,0)); 

    for (int r = 0; r < repetition_count; r++)
    {
        int n = 0;

        LOG(INFO) << "pattern_mode06";
        int ret = lc3010_->pattern_mode06();
        if (DF_SUCCESS != ret)
        {
            frame_status = ret;
            camera_->streamOff();
            return ret;
        }

        lc3010_->start_pattern_sequence();
        LOG(INFO) << "start_pattern_sequence";

        for (int g_i = 0; g_i < 16; g_i++)
        {
            LOG(INFO) << "receiving " << g_i << "th image";
            bool status = camera_->grap((ushort*)img.data);
            LOG(INFO) << "status=" << status;

            if (status)
            {

                cv::cvtColor(img, color_img, cv::COLOR_BayerBG2BGR);

                // bayerToRgb((unsigned short*) img.data, (unsigned short*)  color_img.data);

                std::vector<cv::Mat> channels;
                cv::split(color_img, channels); 

                if(0 == g_i)
                {
                    // cv::Mat blue_img = channels[0].clone();
                    // cv::Mat g_img = channels[1].clone();
                    // cv::Mat r_img = channels[2].clone();

                    // blue_img /= 16;
                    // blue_img.convertTo(blue_img, CV_8U);

                    
                    // g_img /= 16;
                    // g_img.convertTo(g_img, CV_8U);


                    
                    // r_img /= 16;
                    // r_img.convertTo(r_img, CV_8U);

                    // cv::Mat color = color_img.clone();

                    // color /= 16;
                    // color.convertTo(blue_img, CV_8UC3);

                    // cv::imwrite("blue.bmp", blue_img);
                    // cv::imwrite("g_img.bmp", g_img);
                    // cv::imwrite("r_img.bmp", r_img);
                    // cv::imwrite("color.bmp", color);
                }



                cuda_merge_repetition_02_patterns_16((ushort*)channels[0].data, g_i);
            }
            else
            {
                LOG(INFO) << "grad failed!";
                camera_->streamOff(); 

                if (g_i == 0)
                {
                        return DF_ERROR_LOST_TRIGGER;
                }

                frame_status = DF_ERROR_CAMERA_GRAP;
                return DF_ERROR_CAMERA_GRAP;
            }
        }

        /*********************************************************************************************/

        /***********************************************************************************************/
    }

    camera_->streamOff();
    LOG(INFO) << "GXStreamOff";
 

    // setPixelFormat(8);

    cuda_handle_repetition_model06_16(repetition_count);

    cuda_normalize_phase(0);
    LOG(INFO) << "parallel_cuda_unwrap_phase";
    cuda_generate_pointcloud_base_table(); 
    LOG(INFO) << "generate_pointcloud_base_table";

    cuda_copy_depth_from_memory(buff_depth_);
    cuda_copy_pointcloud_from_memory(buff_pointcloud_);

    if (1 == generate_brightness_model_)
    {
        cuda_copy_brightness_from_memory(buff_brightness_);
    }
 
    return frame_status;
}


int Scan3D::captureFrame06RepetitionColor(int repetition_count)
{
        if (hdr_is_shooting_.try_lock()) { lc3010_->disable_hdr_mode(); hdr_is_shooting_.unlock(); }
    LOG(INFO) << "cuda_clear_reconstruct_cache:";
    cuda_clear_reconstruct_cache();
    initCache();

    int frame_status = DF_SUCCESS;
    cuda_clear_repetition_02_patterns();

    int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(8 != bits)
        {
            if(setPixelFormat(8))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }


    if (patterns_sets_num_ < 9)
    {
        return DF_ERROR_LOST_PATTERN_SETS;
    }

    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        frame_status = DF_ERROR_CAMERA_STREAM;
        return DF_ERROR_CAMERA_STREAM;
    }

    // unsigned char *img_ptr= new unsigned char[image_width_*image_height_];
    cv::Mat img(image_height_,image_width_,CV_8U,cv::Scalar(0));
    cv::Mat color_img(image_height_,image_width_,CV_8UC3,cv::Scalar(0,0,0)); 

    for(int r= 0;r< repetition_count;r++)
    {
        int n = 0;
  
        LOG(INFO) << "pattern_mode06";
        int ret = lc3010_->pattern_mode06();
        if(DF_SUCCESS != ret)
        {
            frame_status = ret;
            camera_->streamOff();
            return ret;
        }

        lc3010_->start_pattern_sequence();
        LOG(INFO) << "start_pattern_sequence";

        for (int g_i = 0; g_i < 16; g_i++)
        {
            LOG(INFO) << "receiving " << g_i << "th image";
            bool status = camera_->grap(img.data);
            LOG(INFO) << "status=" << status;

            if (status)
            {

                cv::cvtColor(img, color_img, cv::COLOR_BayerBG2BGR);
                std::vector<cv::Mat> channels;
                cv::split(color_img, channels);

                if (0 == g_i)
                {
                    cuda_copy_brightness_to_memory(channels[0].data);
                }

                cuda_copy_minsw8_pattern_to_memory(channels[0].data, g_i);
                // cuda_copy_pattern_to_memory(img_ptr, i);
                cuda_merge_repetition_02_patterns(g_i);
            }
            else
            {
                LOG(INFO) << "grad failed!";
                camera_->streamOff();
                if (g_i == 0)
                {
                    return DF_ERROR_LOST_TRIGGER;
                } 

                frame_status = DF_ERROR_CAMERA_GRAP;
                return DF_ERROR_CAMERA_GRAP;
            }
        }

        /*********************************************************************************************/

        /***********************************************************************************************/
    }

    camera_->streamOff();
    LOG(INFO) << "GXStreamOff";
 

    cuda_handle_repetition_model06(repetition_count); 
    if (fisher_confidence_val_ > -50)
    {
        phase_monotonicity_filter(fisher_confidence_val_ / 2 + 25);
    }
    cuda_normalize_phase(0); 
    LOG(INFO) << "parallel_cuda_unwrap_phase";
    cuda_generate_pointcloud_base_table();
    // depth_filter(system_config_settings_machine_.Instance().firwmare_param_.radius_filter_r / 1000.);
    LOG(INFO) << "generate_pointcloud_base_table";


    cuda_copy_depth_from_memory(buff_depth_);
    cuda_copy_pointcloud_from_memory(buff_pointcloud_);

    if (1 == generate_brightness_model_)
    { 
        cuda_copy_brightness_from_memory(buff_brightness_);
    }
    // else
    // {

    //     captureTextureImage(generate_brightness_model_, generate_brightness_exposure_,buff_brightness_);
    // }

    return frame_status;
}

int Scan3D::captureFrame06HdrColor()
{

    lc3010_->enable_hdr_mode();std::lock_guard<std::mutex> lock(hdr_is_shooting_);
    LOG(INFO) << "cuda_clear_reconstruct_cache:";
    cuda_clear_reconstruct_cache();
    initCache();

    LOG(INFO)<<"Mixed HDR Exposure Base Confidence:";  
    int frame_status = DF_SUCCESS;

    int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(8 != bits)
        {
            if(setPixelFormat(8))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }



    if (patterns_sets_num_ < 9)
    {
        return DF_ERROR_LOST_PATTERN_SETS;
    }

    LOG(INFO) << "Stream On:";
    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        return DF_ERROR_CAMERA_STREAM;
    }
 
 
    cv::Mat img(image_height_,image_width_,CV_8U,cv::Scalar(0));
    cv::Mat color_img(image_height_,image_width_,CV_8UC3,cv::Scalar(0,0,0)); 
    cv::Mat brightness_img(image_height_,image_width_,CV_8U,cv::Scalar(0));

    for(int hdr_i= 0;hdr_i< hdr_num_;hdr_i++)
    {
        int led_current = led_current_list_[hdr_i];
        int ret;

        LOG(INFO)<<"Set LED: "<<led_current;
 
        float exposure = camera_exposure_list_[hdr_i];

        if (exposure > max_camera_exposure_)
        {
            exposure = max_camera_exposure_;
        }
        else if (exposure < min_camera_exposure_)
        {
            exposure = min_camera_exposure_;
        }

        LOG(INFO) << "Set Camera Exposure Time: " << exposure;

        camera_->setExposure(exposure);

        /***************************************************************************************************/

        ret = lc3010_->pattern_mode06();
        if (DF_SUCCESS != ret)
        {
            camera_->streamOff();
            return ret;
        }

        lc3010_->start_pattern_sequence();

        for (int g_i = 0; g_i < 16; g_i++)
        {
            LOG(INFO) << "grap " << g_i << " image:";
            if (!camera_->grap(img.data))
            {

                camera_->streamOff();
                if (g_i == 0)
                {
                    return DF_ERROR_LOST_TRIGGER;
                }

                return DF_ERROR_CAMERA_GRAP;
            }
            LOG(INFO) << "finished!";



            cv::cvtColor(img, color_img, cv::COLOR_BayerBG2BGR);
            std::vector<cv::Mat> channels;
            cv::split(color_img, channels);

            if (0 == g_i)
            {
                cuda_copy_brightness_to_memory(channels[0].data);
                brightness_img = channels[0].clone();
            } 

            cuda_copy_minsw8_pattern_to_memory(channels[0].data, g_i);

            cuda_handle_minsw8(g_i);
            // copy to gpu

            if (15 == g_i)
            {
                if (fisher_confidence_val_ > -50)
                {
                    phase_monotonicity_filter(fisher_confidence_val_ / 2 + 25);
                }
                cuda_normalize_phase(0);
                cuda_generate_pointcloud_base_table();
                LOG(INFO) << "cuda_generate_pointcloud_base_table";
            }
        }

        /****************************************************************************************************/
 
        cuda_copy_result_to_hdr_color(hdr_i, 0,brightness_img);
    }

    camera_->streamOff();
    cuda_merge_hdr_data(hdr_num_, buff_depth_, buff_brightness_);

    // if (1 != generate_brightness_model_)
    // {
    //     captureTextureImage(generate_brightness_model_, generate_brightness_exposure_, buff_brightness_);
    // }
    /******************************************************************************************************/
    LOG(INFO) << "led_current_: " << led_current_;
    int ret;

    LOG(INFO) << "Set Led: " << led_current_ << "\n";
    LOG(INFO) << "Set Camera Exposure Time: " << camera_exposure_ << "\n";
    camera_->setExposure(camera_exposure_);

    return frame_status;
}


int Scan3D::captureFrame06Hdr()
{
    lc3010_->enable_hdr_mode();std::lock_guard<std::mutex> lock(hdr_is_shooting_);

    LOG(INFO) << "cuda_clear_reconstruct_cache:";
    cuda_clear_reconstruct_cache();
    initCache();


    int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(8 != bits)
        {
            if(setPixelFormat(8))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }


    LOG(INFO)<<"Mixed HDR Exposure Base Confidence:";  
    int frame_status = DF_SUCCESS;

    


    if (patterns_sets_num_ < 9)
    {
        return DF_ERROR_LOST_PATTERN_SETS;
    }

    LOG(INFO) << "Stream On:";
    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        return DF_ERROR_CAMERA_STREAM;
    }
 

    unsigned char *img_ptr = new unsigned char[image_width_ * image_height_];

    for(int hdr_i= 0;hdr_i< hdr_num_;hdr_i++)
    {
        int led_current = led_current_list_[hdr_i];
        int ret;

        LOG(INFO)<<"Set LED: "<<led_current;
 
        float exposure = camera_exposure_list_[hdr_i];

        if (exposure > max_camera_exposure_)
        {
            exposure = max_camera_exposure_;
        }
        else if (exposure < min_camera_exposure_)
        {
            exposure = min_camera_exposure_;
        }

        LOG(INFO) << "Set Camera Exposure Time: " << exposure;

        camera_->setExposure(exposure);

        /***************************************************************************************************/

        ret = lc3010_->pattern_mode06();
        if (DF_SUCCESS != ret)
        {
            camera_->streamOff();
            return ret;
        }

        lc3010_->start_pattern_sequence();

        for (int g_i = 0; g_i< 16; g_i++)
        {
            LOG(INFO) << "grap " << g_i << " image:";
            if (!camera_->grap(img_ptr))
            {

                delete[] img_ptr;
                camera_->streamOff();
                if (g_i == 0)
                {
                        return DF_ERROR_LOST_TRIGGER;
                }
                return DF_ERROR_CAMERA_GRAP;
            }
            LOG(INFO) << "finished!";

            if (0 == g_i)
            {
                cuda_copy_brightness_to_memory(img_ptr);
            }

            cuda_copy_minsw8_pattern_to_memory(img_ptr, g_i);

            cuda_handle_minsw8(g_i);
            // copy to gpu

            if (15 == g_i)
            {
                if (fisher_confidence_val_ > -50)
                {
                    phase_monotonicity_filter(fisher_confidence_val_ / 2 + 25);
                }
                cuda_normalize_phase(0);
                cuda_generate_pointcloud_base_table();
                LOG(INFO) << "cuda_generate_pointcloud_base_table";
            }
        }

        /****************************************************************************************************/

        cuda_copy_result_to_hdr(hdr_i,0);
    }

    delete[] img_ptr;
    camera_->streamOff();
    cuda_merge_hdr_data(hdr_num_, buff_depth_, buff_brightness_);  

    
    // if (1 != generate_brightness_model_)
    // { 
    //     captureTextureImage(generate_brightness_model_, generate_brightness_exposure_, buff_brightness_);
    // }
    /******************************************************************************************************/
    LOG(INFO) << "led_current_: " << led_current_;
    int ret;

    LOG(INFO) << "Set Led: " << led_current_ << "\n"; 
    LOG(INFO) << "Set Camera Exposure Time: " << camera_exposure_ << "\n"; 
    camera_->setExposure(camera_exposure_);

    return frame_status;
 
}



int Scan3D::captureFrame06Hdr_cpu()
{
    lc3010_->enable_hdr_mode(); std::lock_guard<std::mutex> lock(hdr_is_shooting_);

    LOG(INFO) << "cuda_clear_reconstruct_cache:";
    cuda_clear_reconstruct_cache();
    initCache();


    int bits = 0;
    if (DF_SUCCESS == getPixelFormat(bits))
    {
        if (8 != bits)
        {
            if (setPixelFormat(8))
            {
                LOG(INFO) << "set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }


    LOG(INFO) << "Mixed HDR Exposure Base Confidence:";
    int frame_status = DF_SUCCESS;




    if (patterns_sets_num_ < 9)
    {
        return DF_ERROR_LOST_PATTERN_SETS;
    }

    LOG(INFO) << "Stream On:";
    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        return DF_ERROR_CAMERA_STREAM;
    }


    unsigned char* img_ptr = new unsigned char[image_width_ * image_height_];

    std::vector<cv::Mat> hdr_brightness_list;
    std::vector<cv::Mat> hdr_depth_list;
    std::vector<float> hdr_bright_pixel_sum;

    int nr = image_height_;
    int nc = image_width_;
    for (int hdr_i = 0; hdr_i < hdr_num_; hdr_i++)
    {
        int led_current = led_current_list_[hdr_i];
        int ret;

        LOG(INFO) << "Set LED: " << led_current;

        float exposure = camera_exposure_list_[hdr_i];

        if (exposure > max_camera_exposure_)
        {
            exposure = max_camera_exposure_;
        }
        else if (exposure < min_camera_exposure_)
        {
            exposure = min_camera_exposure_;
        }

        LOG(INFO) << "Set Camera Exposure Time: " << exposure;

        camera_->setExposure(exposure);

        /***************************************************************************************************/

        ret = lc3010_->pattern_mode06();
        if (DF_SUCCESS != ret)
        {
            camera_->streamOff();
            return ret;
        }

        lc3010_->start_pattern_sequence();



        int nr = image_height_;
        int nc = image_width_;
        std::vector<cv::Mat> patterns_;
        cv::Mat threshold_map(nr, nc, CV_8U, cv::Scalar(128));
        cv::Mat mask;
        cv::Mat wrap;
        cv::Mat sw_k_map(nr, nc, CV_16U, cv::Scalar(0));
        cv::Mat minsw_map(nr, nc, CV_32F, cv::Scalar(0));
        cv::Mat k2_map;
        cv::Mat unwrap;
        cv::Mat deep_map_table;
        cv::Mat xyz_mat;



        for (int g_i = 0; g_i < 16; g_i++)
        {
            LOG(INFO) << "grap " << g_i << " image:";
            if (!camera_->grap(img_ptr))
            {

                delete[] img_ptr;
                camera_->streamOff();
                if (g_i == 0)
                {
                    return DF_ERROR_LOST_TRIGGER;
                }
                return DF_ERROR_CAMERA_GRAP;
            }

            cv::Mat img(image_height_, image_width_, CV_8UC1, img_ptr);

            if (g_i < 2 || g_i > 7) {
                cv::GaussianBlur(img, img, cv::Size(5, 5), 1, 1);
                std::cout << g_i << "gaussianblur" << std::endl;
            }
            patterns_.push_back(img.clone());
            LOG(INFO) << "finished!";

            cuda_handle_minsw8_cpu(nr, nc, g_i, patterns_, threshold_map, mask, wrap, sw_k_map, minsw_map, k2_map, unwrap);
            // copy to gpu

            if (15 == g_i)
            {
                if (fisher_confidence_val_ > -50)
                {
                    phase_monotonicity_filter(fisher_confidence_val_ / 2 + 25);
                }
                float ver_period = 128;
                unwrap /= ver_period;
                unwrap.convertTo(unwrap, CV_64FC1);






                LookupTableFunction lookup_table_machine_1;
                cv::Mat xL_rotate_x1;
                cv::Mat xL_rotate_y1;
                cv::Mat rectify_R11;
                cv::Mat pattern_mapping1;
                cv::Mat pattern_minimapping1(128, 128, CV_64FC1, cv::Scalar(0));

                bool read_map_ok = lookup_table_machine_1.readTableFloat("./", xL_rotate_x1, xL_rotate_y1, rectify_R11, pattern_mapping1, pattern_minimapping1, image_width_, image_height_);
                float b = sqrt(pow(calib_param_.translation_matrix[0], 2) + pow(calib_param_.translation_matrix[1], 2) + pow(calib_param_.translation_matrix[2], 2));

                lookup_table_machine_1.generate_pointcloud_depth(mask, unwrap, deep_map_table, xyz_mat, xL_rotate_x1, xL_rotate_y1, rectify_R11, pattern_mapping1, b);

                LOG(INFO) << "cuda_generate_pointcloud_base_table";

            }
        }
        /****************************************************************************************************/
        hdr_depth_list.push_back(deep_map_table.clone());
        hdr_brightness_list.push_back(patterns_[0]);
        float val = 0;
        for (int r = 0; r < nr; r++)
        {
            uchar* ptr_b = hdr_brightness_list[hdr_i].ptr<uchar>(r);
            for (int c = 0; c < nc; c++)
            {
                val += ptr_b[c];
            }
        }
        hdr_bright_pixel_sum.push_back(val);
      
    }

    delete[] img_ptr;
    camera_->streamOff();

    //cv::Mat depth;
    //cv::Mat bright;
    cv::Mat depth(image_height_, image_width_, CV_64F, cv::Scalar(0)); // 假设深度图是64位浮点数  
    cv::Mat bright(image_height_, image_width_, CV_8UC1, cv::Scalar(0));
    cuda_merge_hdr_data_cpu(hdr_num_, image_height_,image_width_, hdr_brightness_list,  hdr_depth_list,  hdr_bright_pixel_sum,  depth,  bright);

    for (int r = 0; r < nr; r++)
    {
        double* ptr_w = depth.ptr<double>(r);
        uchar* brights = bright.ptr<uchar>(r);
        for (int c = 0; c < nc; c++)
        {
            buff_depth_[r * depth.cols + c] = static_cast<double>(ptr_w[c]);
            buff_brightness_[r * bright.cols + c] = static_cast<uchar>(brights[c]);
        }
    }


    // if (1 != generate_brightness_model_)
    // { 
    //     captureTextureImage(generate_brightness_model_, generate_brightness_exposure_, buff_brightness_);
    // }
    /******************************************************************************************************/
    LOG(INFO) << "led_current_: " << led_current_;
    int ret;

    LOG(INFO) << "Set Led: " << led_current_ << "\n";
    LOG(INFO) << "Set Camera Exposure Time: " << camera_exposure_ << "\n";
    camera_->setExposure(camera_exposure_);

    return frame_status;

}



int Scan3D::captureFrame06RepetitionMono12(int repetition_count)
{
        if (hdr_is_shooting_.try_lock()) { lc3010_->disable_hdr_mode(); hdr_is_shooting_.unlock(); }
    LOG(INFO) << "cuda_clear_reconstruct_cache:";
    cuda_clear_reconstruct_cache();
    initCache();

    int frame_status = DF_SUCCESS;
    cuda_clear_repetition_02_patterns();

    if (patterns_sets_num_ < 9)
    {
        return DF_ERROR_LOST_PATTERN_SETS;
    }

    int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(12 != bits)
        {
            if(setPixelFormat(12))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }
    // setPixelFormat(12);

    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        frame_status = DF_ERROR_CAMERA_STREAM;
        return DF_ERROR_CAMERA_STREAM;
    }

    unsigned short *img_ptr = new unsigned short[image_width_ * image_height_];

    for (int r = 0; r < repetition_count; r++)
    {
        int n = 0;

        LOG(INFO) << "pattern_mode06";
        int ret = lc3010_->pattern_mode06();
        if (DF_SUCCESS != ret)
        {
            frame_status = ret;
            camera_->streamOff();
            return ret;
        }

        lc3010_->start_pattern_sequence();
        LOG(INFO) << "start_pattern_sequence";

        for (int g_i = 0; g_i < 16; g_i++)
        {
            LOG(INFO) << "receiving " << g_i << "th image";
            bool status = camera_->grap(img_ptr);
            LOG(INFO) << "status=" << status;

            if (status)
            {

                cuda_merge_repetition_02_patterns_16(img_ptr, g_i);
            }
            else
            {
                LOG(INFO) << "grad failed!";
                camera_->streamOff();
                delete[] img_ptr;

                if (g_i == 0)
                {
                        return DF_ERROR_LOST_TRIGGER;
                }

                frame_status = DF_ERROR_CAMERA_GRAP;
                return DF_ERROR_CAMERA_GRAP;
            }
        }

        /*********************************************************************************************/

        /***********************************************************************************************/
    }

    camera_->streamOff();
    LOG(INFO) << "GXStreamOff";

    delete[] img_ptr;

    // setPixelFormat(8);

    cuda_handle_repetition_model06_16(repetition_count);
    if (fisher_confidence_val_ > -50)
    {
        phase_monotonicity_filter(fisher_confidence_val_ / 2 + 25);
    }
    cuda_normalize_phase(0);
    LOG(INFO) << "parallel_cuda_unwrap_phase";
    cuda_generate_pointcloud_base_table(); 
    LOG(INFO) << "generate_pointcloud_base_table";

    cuda_copy_depth_from_memory(buff_depth_);
    cuda_copy_pointcloud_from_memory(buff_pointcloud_);

    if (1 == generate_brightness_model_)
    {
        cuda_copy_brightness_from_memory(buff_brightness_);
    }
 
    return frame_status;
}

int Scan3D::captureFrame06HdrMono12()
{
    lc3010_->enable_hdr_mode();std::lock_guard<std::mutex> lock(hdr_is_shooting_);

    LOG(INFO) << "captureFrame06HdrMono12";
    LOG(INFO) << "cuda_clear_reconstruct_cache:";
    cuda_clear_reconstruct_cache();
    initCache();

    LOG(INFO)<<"Mixed HDR Exposure Base Confidence:";  
    int frame_status = DF_SUCCESS;

     

    if (patterns_sets_num_ < 9)
    {
        return DF_ERROR_LOST_PATTERN_SETS;
    }

    int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(12 != bits)
        {
            if(setPixelFormat(12))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }
    
    // setPixelFormat(12);

    LOG(INFO) << "Stream On:";
    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        return DF_ERROR_CAMERA_STREAM;
    }
 
 
    unsigned short *img_ptr= new unsigned short[image_width_*image_height_];

    for(int hdr_i= 0;hdr_i< hdr_num_;hdr_i++)
    {
        int led_current = led_current_list_[hdr_i];
        int ret;

        LOG(INFO)<<"Set LED: "<<led_current;
 
        float exposure = camera_exposure_list_[hdr_i];

        if (exposure > max_camera_exposure_)
        {
            exposure = max_camera_exposure_;
        }
        else if (exposure < min_camera_exposure_)
        {
            exposure = min_camera_exposure_;
        }

        LOG(INFO) << "Set Camera Exposure Time: " << exposure;

        camera_->setExposure(exposure);

        /***************************************************************************************************/

        ret = lc3010_->pattern_mode06();
        if (DF_SUCCESS != ret)
        {
            camera_->streamOff();
            return ret;
        }

        lc3010_->start_pattern_sequence();

        for (int g_i = 0; g_i< 16; g_i++)
        {
            LOG(INFO) << "grap " << g_i << " image:";
            if (!camera_->grap(img_ptr))
            {

                delete[] img_ptr;
                camera_->streamOff();
                if (g_i == 0)
                {
                        return DF_ERROR_LOST_TRIGGER;
                }
                return DF_ERROR_CAMERA_GRAP;
            }
            LOG(INFO) << "finished!";

  
            cuda_copy_minsw8_pattern_to_memory_16(img_ptr, g_i); 
 
            cuda_handle_minsw8_16(g_i);
 
            // copy to gpu

            if (15 == g_i)
            {
                if (fisher_confidence_val_ > -50)
                {
                    phase_monotonicity_filter(fisher_confidence_val_ / 2 + 25);
                }
                cuda_normalize_phase(0);
                cuda_generate_pointcloud_base_table();
                LOG(INFO) << "cuda_generate_pointcloud_base_table";
            }
        }

        /****************************************************************************************************/
 
        cuda_copy_result_to_hdr(hdr_i,0);
    }

    delete[] img_ptr;
    camera_->streamOff();
    cuda_merge_hdr_data_16(hdr_num_, buff_depth_, buff_brightness_);  

    // setPixelFormat(8); 
 
    /******************************************************************************************************/
    LOG(INFO) << "led_current_: " << led_current_;
    int ret;
    
    LOG(INFO) << "Set Led: " << led_current_ << "\n"; 
    LOG(INFO) << "Set Camera Exposure Time: " << camera_exposure_ << "\n"; 
    camera_->setExposure(camera_exposure_);

    return frame_status;
}

int Scan3D::captureFrame06Mono12()
{
        if (hdr_is_shooting_.try_lock()) { lc3010_->disable_hdr_mode(); hdr_is_shooting_.unlock(); }

    LOG(INFO) << "captureFrame06Mono12";
    LOG(INFO) << "cuda_clear_reconstruct_cache:";
    cuda_clear_reconstruct_cache();
    initCache();

    int frame_status = DF_SUCCESS;
    cuda_clear_repetition_02_patterns();

    if (patterns_sets_num_ < 9)
    {
        return DF_ERROR_LOST_PATTERN_SETS;
    }

    int bits = 0;

    if(DF_SUCCESS == getPixelFormat(bits))
    { 
        if(12 != bits)
        {
            if(setPixelFormat(12))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }

    }
    else
    { 
        LOG(ERROR)<<"Pixel Bits: "<<bits;
    }
    // setPixelFormat(12);


    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        frame_status = DF_ERROR_CAMERA_STREAM;
        return DF_ERROR_CAMERA_STREAM;
    }

    unsigned short *img_ptr = new unsigned short[image_width_ * image_height_];

    LOG(INFO) << "pattern_mode06";
    int ret = lc3010_->pattern_mode06();
    if (DF_SUCCESS != ret)
    {
        frame_status = ret;
        camera_->streamOff();
        return ret;
    }

    lc3010_->start_pattern_sequence();
    LOG(INFO) << "start_pattern_sequence";

    for (int g_i = 0; g_i < 16; g_i++)
    {
        LOG(INFO) << "receiving " << g_i << "th image";
        bool status = camera_->grap(img_ptr);
        LOG(INFO) << "status=" << status;

        if (status)
        {

            cuda_copy_minsw8_pattern_to_memory_16(img_ptr, g_i);

            cuda_handle_minsw8_16(g_i);

            // cuda_handle_model06_16();
            // copy to gpu

            if (15 == g_i)
            {
                if (fisher_confidence_val_ > -50)
                {
                    phase_monotonicity_filter(fisher_confidence_val_ / 2 + 25);
                }
                cuda_normalize_phase(0);
                cuda_generate_pointcloud_base_table();
                LOG(INFO) << "cuda_generate_pointcloud_base_table";
            }
        }
        else
        {
            LOG(INFO) << "grad failed!";
            camera_->streamOff();
            delete[] img_ptr;

            frame_status = DF_ERROR_CAMERA_GRAP;
            return DF_ERROR_CAMERA_GRAP;
        }
    }

        /*********************************************************************************************/

        camera_->streamOff();
        LOG(INFO) << "GXStreamOff";

        delete[] img_ptr;
 
        LOG(INFO) << "generate_pointcloud_base_table";

        // setPixelFormat(8);

        cuda_copy_depth_from_memory(buff_depth_);
        cuda_copy_pointcloud_from_memory(buff_pointcloud_);

        if (1 == generate_brightness_model_)
        {
            cuda_copy_brightness_from_memory(buff_brightness_);
        }
        // else
        // {

        //     captureTextureImage(generate_brightness_model_, generate_brightness_exposure_,buff_brightness_);
        // }

        return frame_status;
}

int Scan3D::captureFrame06()
{
    if (hdr_is_shooting_.try_lock()) { lc3010_->disable_hdr_mode(); hdr_is_shooting_.unlock(); }
    LOG(INFO) << "cuda_clear_reconstruct_cache:";
    cuda_clear_reconstruct_cache();
    initCache();

    int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(8 != bits)
        {
            if(setPixelFormat(8))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }

    int ret = DF_SUCCESS;


 
    
    if(patterns_sets_num_ < 9)
    { 
        return DF_ERROR_LOST_PATTERN_SETS;
    }
    LOG(INFO) << "pattern_mode06 start";

    ret = lc3010_->pattern_mode06();
    if(DF_SUCCESS != ret)
    {
        return ret;
    }
    LOG(INFO) << "pattern_mode06 end";



    LOG(INFO) << "Stream On:";
    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        return DF_ERROR_CAMERA_STREAM;
    }

    lc3010_->start_pattern_sequence();

    unsigned char *img_ptr= new unsigned char[image_width_*image_height_];

    for (int i = 0; i < 16; i++)
    {
        LOG(INFO)<<"grap "<<i<<" image:";
        if (!camera_->grap(img_ptr))
        {

            delete[] img_ptr;
            camera_->streamOff();

            if (i == 0)
            {
                return DF_ERROR_LOST_TRIGGER;
            }

            return DF_ERROR_CAMERA_GRAP;
        }
        LOG(INFO)<<"finished!";

        if(0 == i)
        {
            cuda_copy_brightness_to_memory(img_ptr);
        }


        cuda_copy_minsw8_pattern_to_memory(img_ptr, i);  

        cuda_handle_minsw8(i);
        // copy to gpu
  
        if(15 == i)
        { 
            if (fisher_confidence_val_ > -50)
            {
                phase_monotonicity_filter(fisher_confidence_val_ / 2 + 25);
            }
            cuda_normalize_phase(0); 
            cuda_generate_pointcloud_base_table();
            LOG(INFO) << "cuda_generate_pointcloud_base_table"; 
        } 
   
    }

    delete[] img_ptr;

    camera_->streamOff();
    LOG(INFO) << "Stream Off";

 
    
    cuda_copy_depth_from_memory(buff_depth_);
    cuda_copy_pointcloud_from_memory(buff_pointcloud_);

    if (1 == generate_brightness_model_)
    { 
        cuda_copy_brightness_from_memory(buff_brightness_);
    }
    // else
    // {
    //     captureTextureImage(generate_brightness_model_, generate_brightness_exposure_,buff_brightness_);
    // }

    return DF_SUCCESS;
}

int Scan3D::captureFrame06_cpu()
{
    if (hdr_is_shooting_.try_lock()) { lc3010_->disable_hdr_mode(); hdr_is_shooting_.unlock(); }
    LOG(INFO) << "cuda_clear_reconstruct_cache:";
    cuda_clear_reconstruct_cache();
    initCache();

    int bits = 0;
    if (DF_SUCCESS == getPixelFormat(bits))
    {
        if (8 != bits)
        {
            if (setPixelFormat(8))
            {
                LOG(INFO) << "set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }

    int ret = DF_SUCCESS;




    if (patterns_sets_num_ < 9)
    {
        return DF_ERROR_LOST_PATTERN_SETS;
    }
    LOG(INFO) << "pattern_mode06 start";

    ret = lc3010_->pattern_mode06();
    if (DF_SUCCESS != ret)
    {
        return ret;
    }
    LOG(INFO) << "pattern_mode06 end";



    LOG(INFO) << "Stream On:";
    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        return DF_ERROR_CAMERA_STREAM;
    }

    lc3010_->start_pattern_sequence();

    unsigned char* img_ptr = new unsigned char[image_width_ * image_height_];

    int nr = image_height_;
    int nc = image_width_;
    std::vector<cv::Mat> patterns_;
    cv::Mat threshold_map(nr, nc, CV_8U, cv::Scalar(128));
    cv::Mat mask;
    cv::Mat wrap;
    cv::Mat sw_k_map(nr, nc, CV_16U, cv::Scalar(0));
    cv::Mat minsw_map(nr, nc, CV_32F, cv::Scalar(0));
    cv::Mat k2_map;
    cv::Mat unwrap;
    cv::Mat deep_map_table;
    cv::Mat xyz_mat;

    for (int i = 0; i < 16; i++)
    {
        LOG(INFO) << "grap " << i << " image:";
        if (!camera_->grap(img_ptr))
        {

            delete[] img_ptr;
            camera_->streamOff();

            if (i == 0)
            {
                return DF_ERROR_LOST_TRIGGER;
            }

            return DF_ERROR_CAMERA_GRAP;
        }
        cv::Mat img(image_height_, image_width_, CV_8UC1, img_ptr);

        if (i < 2 || i > 7) {
            cv::GaussianBlur(img, img, cv::Size(5, 5), 1, 1);
            std::cout << i << "gaussianblur" << std::endl;
        }
        patterns_.push_back(img.clone());
        LOG(INFO) << "finished!";

       /* if (0 == i)
        {
            cuda_copy_brightness_to_memory(img_ptr);
        }*/


        //cuda_copy_minsw8_pattern_to_memory(img_ptr, i);

        cuda_handle_minsw8_cpu(nr,nc,i, patterns_, threshold_map,mask,wrap,sw_k_map, minsw_map,k2_map,unwrap);
        // copy to gpu

        if (15 == i)
        {
            if (fisher_confidence_val_ > -50)
            {
                phase_monotonicity_filter(fisher_confidence_val_ / 2 + 25);
            }
            float ver_period = 128;
            unwrap /= ver_period;
            unwrap.convertTo(unwrap, CV_64FC1);

            

     

           
            LookupTableFunction lookup_table_machine_1;
            cv::Mat xL_rotate_x1;
            cv::Mat xL_rotate_y1;
            cv::Mat rectify_R11;
            cv::Mat pattern_mapping1;
            cv::Mat pattern_minimapping1(128, 128, CV_64FC1, cv::Scalar(0));
          
            bool read_map_ok = lookup_table_machine_1.readTableFloat("./", xL_rotate_x1, xL_rotate_y1, rectify_R11, pattern_mapping1, pattern_minimapping1, image_width_, image_height_);
            float b = sqrt(pow(calib_param_.translation_matrix[0], 2) + pow(calib_param_.translation_matrix[1], 2) + pow(calib_param_.translation_matrix[2], 2));
        
            lookup_table_machine_1.generate_pointcloud_depth(mask, unwrap, deep_map_table, xyz_mat, xL_rotate_x1, xL_rotate_y1, rectify_R11, pattern_mapping1,b);
 
            LOG(INFO) << "cuda_generate_pointcloud_base_table";

        }

    }

    delete[] img_ptr;

    camera_->streamOff();
    LOG(INFO) << "Stream Off";



    for (int r = 0; r < nr; r++)
    {
        double* ptr_w = deep_map_table.ptr<double>(r);
        uchar* bright = patterns_[0].ptr<uchar>(r);
        for (int c = 0; c < nc; c++)
        {
            buff_depth_[r * deep_map_table.cols + c] = static_cast<float>(ptr_w[c]);
            buff_brightness_[r * patterns_[0].cols + c] = static_cast<uchar>(bright[c]);
        }
    }



    //cuda_copy_depth_from_memory(buff_depth_);
    //cuda_copy_pointcloud_from_memory(buff_pointcloud_);

   /* if (1 == generate_brightness_model_)
    {
        cuda_copy_brightness_from_memory(buff_brightness_);
    }*/
    // else
    // {
    //     captureTextureImage(generate_brightness_model_, generate_brightness_exposure_,buff_brightness_);
    // }

    return DF_SUCCESS;
}



int Scan3D::captureFrame06Color()
{
        if (hdr_is_shooting_.try_lock()) { lc3010_->disable_hdr_mode(); hdr_is_shooting_.unlock(); }
    LOG(INFO) << "cuda_clear_reconstruct_cache:";
    cuda_clear_reconstruct_cache();
    initCache();

    int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(8 != bits)
        {
            if(setPixelFormat(8))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }



    int ret = DF_SUCCESS;
 
    
    if(patterns_sets_num_ < 9)
    { 
        return DF_ERROR_LOST_PATTERN_SETS;
    }

    ret = lc3010_->pattern_mode06();
    if(DF_SUCCESS != ret)
    {
        return ret;
    }



    LOG(INFO) << "Stream On:";
    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        return DF_ERROR_CAMERA_STREAM;
    }

    lc3010_->start_pattern_sequence();


    // unsigned char *img_ptr= new unsigned char[image_width_*image_height_];

    cv::Mat color_img(image_height_,image_width_,CV_8UC3,cv::Scalar(0,0,0));
    cv::Mat img(image_height_,image_width_,CV_8U,cv::Scalar(0));

    for (int i = 0; i < 16; i++)
    {
        LOG(INFO)<<"grap "<<i<<" image:";
        if (!camera_->grap(img.data))
        { 
            camera_->streamOff();
            if (i == 0)
            {
                return DF_ERROR_LOST_TRIGGER;
            }
            return DF_ERROR_CAMERA_GRAP;
        }
        LOG(INFO)<<"finished!";



        cv::cvtColor(img,color_img,cv::COLOR_BayerBG2BGR);

        std::vector<cv::Mat> channels;
        cv::split(color_img,channels);


        if(0 == i)
        {
            cuda_copy_brightness_to_memory(channels[0].data);
        }

        cuda_copy_minsw8_pattern_to_memory(channels[0].data, i);  

        cuda_handle_minsw8(i);
        // copy to gpu
  
        if(15 == i)
        { 
            if (fisher_confidence_val_ > -50)
            {
                phase_monotonicity_filter(fisher_confidence_val_ / 2 + 25);
            }
            cuda_normalize_phase(0); 
            cuda_generate_pointcloud_base_table();
            LOG(INFO) << "cuda_generate_pointcloud_base_table"; 
        } 
   
    }
 

    camera_->streamOff();
    LOG(INFO) << "Stream Off";
 
    
    cuda_copy_depth_from_memory(buff_depth_);
    cuda_copy_pointcloud_from_memory(buff_pointcloud_);

    if (1 == generate_brightness_model_)
    { 
        cuda_copy_brightness_from_memory(buff_brightness_);
    }
    // else
    // {
    //     captureTextureImage(generate_brightness_model_, generate_brightness_exposure_,buff_brightness_);
    // }

    return DF_SUCCESS;
}


int Scan3D::captureHdrBrightness(unsigned char* buff)
{

}

void Scan3D::mergeBrightness()
{
    cuda_merge_brigntness(hdr_num_, buff_brightness_);  
}

int Scan3D::captureFrame04HdrBaseConfidence()
{
    lc3010_->enable_hdr_mode();std::lock_guard<std::mutex> lock(hdr_is_shooting_);
    cuda_clear_reconstruct_cache();
    initCache();

    int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(8 != bits)
        {
            if(setPixelFormat(8))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }


    LOG(INFO)<<"Mixed HDR Exposure Base Confidence:";  
    int frame_status = DF_SUCCESS;
 
    for(int i= 0;i< hdr_num_;i++)
    {
        int led_current = led_current_list_[i];
        int ret;

        LOG(INFO)<<"Set LED: "<<led_current;
 
        float exposure = camera_exposure_list_[i];

        if (exposure > max_camera_exposure_)
        {
            exposure = max_camera_exposure_;
        }
        else if (exposure < min_camera_exposure_)
        {
            exposure = min_camera_exposure_;
        }

        LOG(INFO) << "Set Camera Exposure Time: " << exposure;

        camera_->setExposure(exposure);

        ret = captureFrame04BaseConfidence();


        if (DF_SUCCESS != ret)
        {
            LOG(ERROR) << "captureFrame04BaseConfidence code: " << ret;
            frame_status = ret;
            return ret;
        }
        cuda_copy_result_to_hdr(i,18);
    }
 


    cuda_merge_hdr_data(hdr_num_, buff_depth_, buff_brightness_);  

    
    // if (1 != generate_brightness_model_)
    // { 
    //     captureTextureImage(generate_brightness_model_, generate_brightness_exposure_, buff_brightness_);
    // }
    /******************************************************************************************************/
 
    int ret;
    
    LOG(INFO) << "Set Led: " << led_current_ << "\n"; 
    LOG(INFO) << "Set Camera Exposure Time: " << camera_exposure_ << "\n"; 
    camera_->setExposure(camera_exposure_);

    return frame_status;
}

bool Scan3D::captureFrame04Repetition01(int repetition_count)
{
        if (hdr_is_shooting_.try_lock()) { lc3010_->disable_hdr_mode(); hdr_is_shooting_.unlock(); }

    lc3010_->pattern_mode04_repetition(repetition_count);
    LOG(INFO) << "Stream On:";
    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        return false;
    }

    lc3010_->start_pattern_sequence();

    unsigned char *img_ptr = new unsigned char[image_width_ * image_height_];

    int capture_count = 19 + 6 * (repetition_count - 1);

    for (int i = 0; i < capture_count; i++)
    {
        LOG(INFO) << "receiving " << i << "th image"; 
        if (!camera_->grap(img_ptr))
        {
            
            delete[] img_ptr; 
            camera_->streamOff();
 
            return false;
        } 
 

        int sync_serial_num = i;

        if (i < 12)
        {
            sync_serial_num = i;
            cuda_copy_pattern_to_memory(img_ptr, i);
        }
        else if (i > 11 && i < 12 + 6 * repetition_count)
        {

            if (12 == i)
            {
                sync_serial_num = 12;
            }
            else
            {
                sync_serial_num = 13;
            }

            cuda_copy_repetition_pattern_to_memory(img_ptr, i - 12);
            cuda_merge_repetition_patterns(i - 12); 
            LOG(INFO) << "repetition " << i - 12 << "th image";
        }
        else
        {

            sync_serial_num = i - 6 * (repetition_count - 1);
            cuda_copy_pattern_to_memory(img_ptr, sync_serial_num);
            if (18 == sync_serial_num)
            {
                cuda_copy_brightness_to_memory(img_ptr);
            }
        }

        // copy to gpu
        switch (sync_serial_num)
        {
        case 4:
        {
            cuda_compute_phase_shift(0);
        }
        break;
        case 8:
        {
            cuda_compute_phase_shift(1);
        }
        break;
        case 10:
        {
            cuda_unwrap_phase_shift(1);
        }
        break;
        case 12:
        {
            cuda_compute_phase_shift(2);
            cuda_unwrap_phase_shift(2);
        }
        break;
        // case 15:
        // {
        // }
        // break;
        case 18:
        {

            cuda_compute_merge_phase(repetition_count);
            cudaDeviceSynchronize();
            // parallel_cuda_compute_phase(3);
            cuda_unwrap_phase_shift(3);

            cuda_normalize_phase(0);
            cuda_generate_pointcloud_base_table();
            //  cudaDeviceSynchronize();
            LOG(INFO) << "generate_pointcloud_base_table";
        }
        break;

            break;

        default:
            break;
        }
    }



    delete[] img_ptr;

    camera_->streamOff();
    LOG(INFO) << "Stream Off";
  
    cuda_copy_depth_from_memory(buff_depth_);
    cuda_copy_pointcloud_from_memory(buff_pointcloud_);

    if (1 == generate_brightness_model_)
    { 
        cuda_copy_brightness_from_memory(buff_brightness_);
    }
    else
    {

        captureTextureImage(generate_brightness_model_, generate_brightness_exposure_,buff_brightness_);
    }


    return true;
}


int Scan3D::captureFrame04Repetition02BaseConfidence(int repetition_count)
{
        if (hdr_is_shooting_.try_lock()) { lc3010_->disable_hdr_mode(); hdr_is_shooting_.unlock(); }
    cuda_clear_repetition_02_patterns();
    cuda_clear_reconstruct_cache();
    initCache();

    int bits = 0;
    if(DF_SUCCESS == getPixelFormat(bits))
    {
        if(8 != bits)
        {
            if(setPixelFormat(8))
            {
                LOG(INFO)<<"set camera pixel format error!";
                return DF_FAILED;
            }
        }
    }


    int frame_status = DF_SUCCESS;


    unsigned char *img_ptr= new unsigned char[image_width_*image_height_];

    for(int r= 0;r< repetition_count;r++)
    {
        int n = 0;
  
        LOG(INFO) << "pattern_mode04";
        int ret = lc3010_->pattern_mode04();
        if(DF_SUCCESS != ret)
        {
            frame_status = ret;
            return ret;
        }
        if (!camera_->streamOn())
        {
            LOG(INFO) << "Stream On Error";
            frame_status = DF_ERROR_CAMERA_STREAM;
            return DF_ERROR_CAMERA_STREAM;
        }

        lc3010_->start_pattern_sequence();
        LOG(INFO) << "start_pattern_sequence";

        for (int i = 0; i < 19; i++)
        {
            LOG(INFO) << "receiving " << i << "th image";
            bool status = camera_->grap(img_ptr);
            LOG(INFO) << "status=" << status;

            if (status)
            {

                cuda_copy_pattern_to_memory(img_ptr, i);
                cuda_merge_repetition_02_patterns(i);
            }
            else
            {
                LOG(INFO) << "grad failed!";
                camera_->streamOff();
                delete[] img_ptr;

                if (i == 0)
                {
                    return DF_ERROR_LOST_TRIGGER;
                }

                frame_status = DF_ERROR_CAMERA_GRAP;
                return DF_ERROR_CAMERA_GRAP;
            }
        }

        /*********************************************************************************************/
        camera_->streamOff();
        LOG(INFO) << "GXStreamOff";
        /***********************************************************************************************/
    }

    delete[] img_ptr;

    cuda_compute_merge_repetition_02_phase(repetition_count,1);
    LOG(INFO) << "parallel_cuda_compute_mergerepetition_02_phase";
    cuda_unwrap_phase_shift_base_fisher_confidence(1);
    cuda_unwrap_phase_shift_base_fisher_confidence(2);
    if (1 == system_config_settings_machine_.Instance().firwmare_param_.use_gray_rectify)
    {
        cv::Mat convolution_kernal = cv::getGaussianKernel(system_config_settings_machine_.Instance().firwmare_param_.gray_rectify_r, system_config_settings_machine_.Instance().firwmare_param_.gray_rectify_sigma * 0.02, CV_32F);
        convolution_kernal = convolution_kernal * convolution_kernal.t();
        cuda_copy_convolution_kernal_to_memory((float*)convolution_kernal.data, system_config_settings_machine_.Instance().firwmare_param_.gray_rectify_r);
        cuda_rectify_six_step_pattern_phase(1, system_config_settings_machine_.Instance().firwmare_param_.gray_rectify_r);
    }
    cuda_unwrap_phase_shift_base_fisher_confidence(3);

    LOG(INFO) << "fisher_confidence_val： " << fisher_confidence_val_;
    if (-50 < fisher_confidence_val_)
    {
        fisher_filter(fisher_confidence_val_);
    }

    cuda_normalize_phase(0);

    LOG(INFO) << "parallel_cuda_unwrap_phase";
    cuda_generate_pointcloud_base_table();
    // depth_filter(system_config_settings_machine_.Instance().firwmare_param_.radius_filter_r / 1000.);
    LOG(INFO) << "generate_pointcloud_base_table";


    cuda_copy_depth_from_memory(buff_depth_);
    cuda_copy_pointcloud_from_memory(buff_pointcloud_);

    // if (1 == generate_brightness_model_)
    // { 
        cuda_copy_brightness_from_memory(buff_brightness_);
    // }
    // else
    // {

    //     captureTextureImage(generate_brightness_model_, generate_brightness_exposure_,buff_brightness_);
    // }

    return frame_status;
}

// bool Scan3D::captureFrame04Repetition02(int repetition_count)
// {
 
//     cuda_clear_repetition_02_patterns();

//     unsigned char *img_ptr= new unsigned char[image_width_*image_height_];

//     for(int r= 0;r< repetition_count;r++)
//     {
//         int n = 0;
  
//         LOG(INFO) << "pattern_mode04";
//         lc3010_->pattern_mode04();

//         if (!camera_->streamOn())
//         {
//             LOG(INFO) << "Stream On Error";
//             return false;
//         }

//         lc3010_->start_pattern_sequence();
//         LOG(INFO) << "start_pattern_sequence";

//         for (int i = 0; i < 19; i++)
//         {
//             LOG(INFO) << "receiving " << i << "th image";
//             bool status = camera_->grap(img_ptr);
//             LOG(INFO) << "status=" << status;

//             if (status)
//             {

//                 cuda_copy_pattern_to_memory(img_ptr, i);
//                 cuda_merge_repetition_02_patterns(i);
//             }
//             else
//             {
//                 LOG(INFO) << "grad failed!";
//                 camera_->streamOff();
//                 delete[] img_ptr;
//                 if (i == 0)
//                 {
//                         return DF_ERROR_LOST_TRIGGER;
//                 }
//                 return false;
//             }
//         }

//         /*********************************************************************************************/
//         camera_->streamOff();
//         LOG(INFO) << "GXStreamOff";
//         /***********************************************************************************************/
//     }

//     delete[] img_ptr;

//     cuda_compute_merge_repetition_02_phase(repetition_count,1);
//     LOG(INFO) << "parallel_cuda_compute_mergerepetition_02_phase";
//     cuda_unwrap_phase_shift(1);
//     cuda_unwrap_phase_shift(2);
//     cuda_unwrap_phase_shift(3);

//     cuda_normalize_phase(0);
//     LOG(INFO) << "parallel_cuda_unwrap_phase";
//     cuda_generate_pointcloud_base_table();
//     LOG(INFO) << "generate_pointcloud_base_table";


//     cuda_copy_depth_from_memory(buff_depth_);
//     cuda_copy_pointcloud_from_memory(buff_pointcloud_);

//     if (1 == generate_brightness_model_)
//     { 
//         cuda_copy_brightness_from_memory(buff_brightness_);
//     }
//     else
//     {

//         captureTextureImage(generate_brightness_model_, generate_brightness_exposure_,buff_brightness_);
//     }

//     return true;
// }


bool Scan3D::captureFrame05()
{

        if (hdr_is_shooting_.try_lock()) { lc3010_->disable_hdr_mode(); hdr_is_shooting_.unlock(); }
    lc3010_->pattern_mode04();
    LOG(INFO) << "Stream On:";
    if (!camera_->streamOn())
    {
        LOG(INFO) << "Stream On Error";
        return false;
    }

    lc3010_->start_pattern_sequence();

    unsigned char *img_ptr= new unsigned char[image_width_*image_height_];

    for (int i = 0; i < 19; i++)
    {
        LOG(INFO)<<"grap "<<i<<" image:";
        if (!camera_->grap(img_ptr))
        {
            
            delete[] img_ptr; 
            camera_->streamOff();
            if (i == 0)
            {
                return DF_ERROR_LOST_TRIGGER;
            }
            return false;
        }
        LOG(INFO)<<"finished!";

        if(18 == i)
        {
            cuda_copy_brightness_to_memory(img_ptr);
        }

        cuda_copy_pattern_to_memory(img_ptr, i);

        // copy to gpu
        switch (i)
        {
        case 4:
        {
        LOG(INFO)<<"cuda_compute_phase_shift:";
            cuda_compute_phase_shift(0); 
        }
        break;
        case 8:
        {
            cuda_compute_phase_shift(1);
        }
        break;
        case 10:
        {
            cuda_unwrap_phase_shift(1);
        }
        break;
        case 12:
        {
            cuda_compute_phase_shift(2);
        }
        break;
        case 15:
        {
            cuda_unwrap_phase_shift(2);
        }
        break;
        case 18:
        {


            cuda_compute_phase_shift(3);
            cuda_unwrap_phase_shift(3);
            cuda_normalize_phase(0);

            cuda_generate_pointcloud_base_minitable();
            //  cudaDeviceSynchronize();
            LOG(INFO) << "cuda_generate_pointcloud_base_table";
        }

        default:
            break;
        }
    }

    delete[] img_ptr;

    camera_->streamOff();
    LOG(INFO) << "Stream Off";

    if(1 == system_config_settings_machine_.Instance().firwmare_param_.use_radius_filter)
    {
        float r = system_config_settings_machine_.Instance().firwmare_param_.radius_filter_r;
        int num = system_config_settings_machine_.Instance().firwmare_param_.radius_filter_threshold_num;
        cuda_remove_points_base_radius_filter(0.5,r,num);
    }

    
    cuda_copy_depth_from_memory(buff_depth_);
    cuda_copy_pointcloud_from_memory(buff_pointcloud_);

    if (1 == generate_brightness_model_)
    { 
        cuda_copy_brightness_from_memory(buff_brightness_);
    }
    else
    {

        captureTextureImage(generate_brightness_model_, generate_brightness_exposure_,buff_brightness_);
    }

 
    return true;
}


bool Scan3D::captureFrame03()
{

        if (hdr_is_shooting_.try_lock()) { lc3010_->disable_hdr_mode(); hdr_is_shooting_.unlock(); }
    // lc3010_->pattern_mode03();
    // LOG(INFO) << "Stream On:";
    // if (!camera_->streamOn())
    // {
    //     LOG(INFO) << "Stream On Error";
    //     return false;
    // }

    // lc3010_->start_pattern_sequence();

    // unsigned char *img_ptr = new unsigned char[image_width_ * image_height_];

    // for (int i = 0; i < 31; i++)
    // {

    //     LOG(INFO) << "grap " << i << " image:";
    //     if (!camera_->grap(img_ptr))
    //     {

    //         delete[] img_ptr;
    //         camera_->streamOff();
    //         return false;
    //     }
    //     LOG(INFO) << "finished!";

    //     parallel_cuda_copy_signal_patterns(img_ptr, i);

    //     // copy to gpu
    //     switch (i)
    //     {
    //     case 4:
    //     {
    //         parallel_cuda_compute_phase(0);
    //     }
    //     break;
    //     case 8:
    //     {
    //         parallel_cuda_compute_phase(1);
    //         parallel_cuda_unwrap_phase(1);
    //     }
    //     break;
    //     case 12:
    //     {
    //         parallel_cuda_compute_phase(2);
    //         parallel_cuda_unwrap_phase(2);
    //     }
    //     break;
    //     case 18:
    //     {
    //         parallel_cuda_compute_phase(3);
    //         parallel_cuda_unwrap_phase(3);
    //     }
    //     break;
    //     case 21:
    //     {
    //         parallel_cuda_compute_phase(4);
    //     }
    //     break;
    //     case 25:
    //     {
    //         parallel_cuda_compute_phase(5);
    //         parallel_cuda_unwrap_phase(5);
    //     }
    //     break;
    //     case 30:
    //     {
    //         parallel_cuda_compute_phase(6);
    //         parallel_cuda_unwrap_phase(6);

    //         // cudaDeviceSynchronize();
    //         parallel_cuda_reconstruct();
    //     }
    //     break;

    //     default:
    //         break;
    //     }
    // }

    // delete[] img_ptr;

    // camera_->streamOff();
    // LOG(INFO) << "Stream Off";

    // parallel_cuda_copy_result_from_gpu(buff_depth_, buff_brightness_);
    // // reconstruct_copy_pointcloud_from_cuda_memory(buff_pointcloud_);

    // if (1 != generate_brightness_model_)
    // {
    //     captureTextureImage(generate_brightness_model_, generate_brightness_exposure_, buff_brightness_);
    // }

    return true;
}


bool Scan3D::captureFrame01()
{ 
        if (hdr_is_shooting_.try_lock()) { lc3010_->disable_hdr_mode(); hdr_is_shooting_.unlock(); }
    // int buffer_size = image_width_*image_height_*24;
    // unsigned char* buffer = new unsigned char[buffer_size];

    // if (!captureRaw01(buffer))
    // { 
    //     LOG(INFO) << "capture Raw 01 Failed!"; 
    //     delete[] buffer;
    // }
 
    // std::vector<unsigned char*> patterns_ptr_list;
    // for(int i=0; i<24; i++)
    // {
	//     patterns_ptr_list.push_back(((unsigned char*)(buffer+i*image_width_*image_height_)));
    // }
  
    // cuda_get_frame_base_24(patterns_ptr_list, buff_depth_,buff_brightness_);


    // delete[] buffer;
    return true;
}


bool Scan3D::testCaptureFrame01(unsigned char* buffer)
{
        if (hdr_is_shooting_.try_lock()) { lc3010_->disable_hdr_mode(); hdr_is_shooting_.unlock(); }
    // std::vector<unsigned char*> patterns_ptr_list;
    // for(int i=0; i<24; i++)
    // {
	//     patterns_ptr_list.push_back(((unsigned char*)(buffer+i*image_width_*image_height_)));
    // }
  
    // cuda_get_frame_base_24(patterns_ptr_list, buff_depth_,buff_brightness_);
    return true;
}
/***********************************************************************************************************************/

bool Scan3D::readCalibParam()
{
    std::ifstream ifile; 
    ifile.open("calib_param.txt");

    if(!ifile.is_open())
    {
        return false;
    }

    int n_params = sizeof(calib_param_)/sizeof(float);
    for(int i=0; i<n_params; i++)
    {
	    ifile>>(((float*)(&calib_param_))[i]); 
    }
    ifile.close();
    return true;
}


bool Scan3D::loadCalibData()
{

    if (!readCalibParam())
    {
        LOG(INFO) << "Read Calib Param Error!";
        return false;
    }
    else
    {

        LOG(INFO) << "cuda_copy_calib_data:";
        cuda_copy_calib_data(calib_param_.camera_intrinsic,
                             calib_param_.projector_intrinsic,
                             calib_param_.camera_distortion,
                             calib_param_.projector_distortion,
                             calib_param_.rotation_matrix,
                             calib_param_.translation_matrix);

        LOG(INFO) << "generate undistort table:";

        int nr = image_height_;
        int nc = image_width_;



        cv::Mat undistort_map_x(nr, nc, CV_32F, cv::Scalar(0));
        cv::Mat undistort_map_y(nr, nc, CV_32F, cv::Scalar(0));

        float camera_fx = calib_param_.camera_intrinsic[0];
        float camera_fy = calib_param_.camera_intrinsic[4];

        float camera_cx = calib_param_.camera_intrinsic[2];
        float camera_cy = calib_param_.camera_intrinsic[5];

        float k1 = calib_param_.camera_distortion[0];
        float k2 = calib_param_.camera_distortion[1];
        float p1 = calib_param_.camera_distortion[2];
        float p2 = calib_param_.camera_distortion[3];
        float k3 = calib_param_.camera_distortion[4];

        for (int r = 0; r < nr; r++)
        {

            for (int c = 0; c < nc; c++)
            {
                double undistort_x = c;
                double undistort_y = r;

                int offset = r * nc + c;

                undistortPoint(c, r, camera_fx, camera_fy,
                               camera_cx, camera_cy, k1, k2, k3, p1, p2, undistort_x, undistort_y);

                undistort_map_x.at<float>(r,c) = undistort_x;
                undistort_map_y.at<float>(r,c)  =  undistort_y;
            }
        }

        coud_copy_undistort_table_to_memory((float*)undistort_map_x.data,(float*)undistort_map_y.data);
    }

    LookupTableFunction lookup_table_machine_; 
    MiniLookupTableFunction minilookup_table_machine_;

	lookup_table_machine_.setCalibData(calib_param_);
    minilookup_table_machine_.setCalibData(calib_param_);

    LOG(INFO)<<"start read table:";
    
    cv::Mat xL_rotate_x;
    cv::Mat xL_rotate_y;
    cv::Mat rectify_R1;
    cv::Mat pattern_mapping;
    cv::Mat pattern_minimapping(128,128,CV_32F,cv::Scalar(0));

    bool read_map_ok = lookup_table_machine_.readTableFloat("./", xL_rotate_x, xL_rotate_y, rectify_R1, pattern_mapping,pattern_minimapping,image_width_,image_height_);

    

  

 
    if(read_map_ok)
    {  
        LOG(INFO)<<"read table finished!";
	    cv::Mat R1_t = rectify_R1.t();
        xL_rotate_x.convertTo(xL_rotate_x, CV_32F);
        xL_rotate_y.convertTo(xL_rotate_y, CV_32F);
        R1_t.convertTo(R1_t, CV_32F);
        pattern_mapping.convertTo(pattern_mapping, CV_32F);
        pattern_minimapping.convertTo(pattern_minimapping, CV_32F); 

        float b = sqrt(pow(calib_param_.translation_matrix[0], 2) + pow(calib_param_.translation_matrix[1], 2) + pow(calib_param_.translation_matrix[2], 2));
  
        std::cout << "b"<<b << std::endl;
        LOG(INFO)<<"start copy table:";
        cuda_copy_talbe_to_memory((float*)pattern_mapping.data,(float*)pattern_minimapping.data, (float*)xL_rotate_x.data,(float*)xL_rotate_y.data,(float*)R1_t.data,b);
        LOG(INFO)<<"copy finished!";
    }

    return true;
}

void Scan3D::copyBrightnessData(unsigned char* &ptr)
{ 
	memcpy(ptr, buff_brightness_, sizeof(unsigned char)*image_height_*image_width_); 
}

void Scan3D::copyDepthData(float* &ptr)
{ 
	memcpy(ptr, buff_depth_, sizeof(float)*image_height_*image_width_);
} 

void Scan3D::copyPointcloudData(float* &ptr)
{ 
    memcpy(ptr, buff_pointcloud_, sizeof(float) * image_height_ * image_width_ * 3);
}

void Scan3D::getCameraResolution(int &width, int &height)
{
    width = image_width_;
    height = image_height_;
}



int Scan3D::getPixelFormat(int &bit)
{
    int ret = DF_SUCCESS;

    int val = 0;

    bool ok = camera_->getPixelFormat(val);


    if(!ok)
    {
        // bit = 8;
        LOG(ERROR)<<"getPixelFormat";
        return DF_FAILED;
    }

    bit = val;

    return DF_SUCCESS;
    
}

int Scan3D::setPixelFormat(int bit)
{
    int ret = DF_SUCCESS;

    switch (bit)
    {
    case 8:
    {

        
        setParamConfidence(system_config_settings_machine_.Instance().firwmare_param_.confidence);
        
        camera_->setPixelFormat(8); 

        LOG(INFO) << "min_camera_exposure_mono8_: " << min_camera_exposure_mono8_;
        lc3010_->set_camera_min_exposure(min_camera_exposure_mono8_);

        if (!camera_->setExposure(camera_exposure_))
        {
            return false;
        }
        lc3010_->set_camera_exposure(camera_exposure_);
 
    }
    break;

    case 12:
    {
        

        setParamConfidence(system_config_settings_machine_.Instance().firwmare_param_.confidence*16);

        camera_->setPixelFormat(12); 
  
        LOG(INFO) << "min_camera_exposure_mono12_: " << min_camera_exposure_mono12_;
        lc3010_->set_camera_min_exposure(min_camera_exposure_mono12_);

        if (!camera_->setExposure(camera_exposure_))
        {
            return false;
        }
        lc3010_->set_camera_exposure(camera_exposure_);
    }
    break;

    default:
        ret = DF_FAILED;
        break;
    }

    return ret;
}

void Scan3D::getCameraPixelType(XemaPixelType &type) 
{ 
    camera_->getPixelType(type);
}

void Scan3D::removeOutlierBaseRadiusFilter()
{
    if(1 == system_config_settings_machine_.Instance().firwmare_param_.use_radius_filter)
    {
        float r = system_config_settings_machine_.Instance().firwmare_param_.radius_filter_r;
        int num = system_config_settings_machine_.Instance().firwmare_param_.radius_filter_threshold_num;
        LOG(INFO)<<"radius_filter_r: "<<r;
        LOG(INFO)<<"num: "<<num; 

        cuda_remove_points_base_radius_filter(0.05,r,num);

        cuda_copy_depth_from_memory(buff_depth_); 
        LOG(INFO)<<"removal finished!";
    }
}

void Scan3D::removeOutlierBaseDepthFilter()
{
    if(1 == system_config_settings_machine_.Instance().firwmare_param_.use_depth_filter)
    {
        float depth_threshold = system_config_settings_machine_.Instance().firwmare_param_.depth_filter_threshold * (-0.128) + 13.;
        LOG(INFO)<<"depth_filter_threshold: "<<depth_threshold;

        depth_filter(depth_threshold / 1000.);

        cuda_copy_depth_from_memory(buff_depth_);
    }
}

CameraGalaxy* Scan3D::getCameraDevice()
{
    return camera_;
}


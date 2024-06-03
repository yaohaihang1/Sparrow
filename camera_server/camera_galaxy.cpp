#include "camera_galaxy.h"
#include<vector>
#include<fstream>
#include"set_calib_looktable.h"
#include"critical zone.h"
#include "easylogging++.h"

#define Write_Pattern_Order             0x98
#define Read_Pattern_Order              0x99
CameraGalaxy::CameraGalaxy()
{
}
CameraGalaxy::~CameraGalaxy()
{
}


bool CameraGalaxy::trigger_software()
{
    GX_STATUS status = GX_STATUS_SUCCESS;
    // 发送软触发命令
    status = GXSendCommand(hDevice_, GX_COMMAND_TRIGGER_SOFTWARE);
    if (status != GX_STATUS_SUCCESS)
    {
        return false;
    }
    return true;
}

bool CameraGalaxy::streamOn() {

    GX_STATUS status = GX_STATUS_SUCCESS;
    status = GXSendCommand(hDevice_, GX_COMMAND_ACQUISITION_START);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "设置开始采集失败！" << std::endl;

    }
    /*status = GXSendCommand(hDevice_, GX_TRIGGER_SOURCE_LINE0);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "设置line0失败！" << std::endl;

    }*/
    return true;
}

bool CameraGalaxy::streamOff() {

    GX_STATUS status = GX_STATUS_SUCCESS;
    status = GXSendCommand(hDevice_, GX_COMMAND_ACQUISITION_STOP);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "设置停止采集失败！" << std::endl;

    }
    return true;
}


bool CameraGalaxy::grap(unsigned char* buf)
{
    if (pixel_format_ != 8)
    {
        LOG(INFO) << "set pixel format";
   
        return false;
    }

    int64_t nPayLoadSize = 0;
    GX_STATUS status = GX_STATUS_SUCCESS;
    status = GXGetInt(hDevice_, GX_INT_PAYLOAD_SIZE, &nPayLoadSize);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "获取数据大小失败！" << std::endl;
    }

    pFrameData.pImgBuf = malloc((size_t)nPayLoadSize);

    if (GXGetImage(hDevice_, &pFrameData, 1000) != GX_STATUS_SUCCESS)
    {
        return false;
    }

    if (pFrameData.nStatus == GX_FRAME_STATUS_SUCCESS && pFrameData.pImgBuf != nullptr)
    {
        memcpy(buf, pFrameData.pImgBuf, pFrameData.nImgSize);
        free(pFrameData.pImgBuf);
    }
    else
    {
        free(pFrameData.pImgBuf);
        return false;
    }

    return true;
}

bool CameraGalaxy::grap(unsigned short* buf)
{
    if (pixel_format_ != 12)
    {
        LOG(INFO) << "set pixel format";
        return false;
    }

    int64_t nPayLoadSize = 0;
    GX_STATUS status = GX_STATUS_SUCCESS;
    status = GXGetInt(hDevice_, GX_INT_PAYLOAD_SIZE, &nPayLoadSize);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "获取数据大小失败！" << std::endl;
    }

    pFrameData.pImgBuf = malloc((size_t)nPayLoadSize);

    if (GXGetImage(hDevice_, &pFrameData, 1000) != GX_STATUS_SUCCESS)
    {
        return false;
    }

    if (pFrameData.nStatus == GX_FRAME_STATUS_SUCCESS && pFrameData.pImgBuf != nullptr)
    {
        memcpy(buf, pFrameData.pImgBuf, pFrameData.nImgSize);
        free(pFrameData.pImgBuf);
    }
    else
    {
        free(pFrameData.pImgBuf);
        return false;
    }

    return true;
}

bool CameraGalaxy::setPixelFormat(int val)
{
    
    LOG(INFO) << "set pixel format:"<<val;
    GXSendCommand(hDevice_, GX_COMMAND_ACQUISITION_STOP);

    switch (val)
    {
    case 8:
        GXSetEnum(hDevice_, GX_ENUM_PIXEL_FORMAT, GX_PIXEL_FORMAT_MONO8);
        pixel_format_ = val;
        break;
    case 12:
        GXSetEnum(hDevice_, GX_ENUM_PIXEL_FORMAT, GX_PIXEL_FORMAT_MONO10);
        pixel_format_ = val;
        break;
    default:
        break;
    }

    return true;
}

bool CameraGalaxy::getPixelFormat(int& val)
{
    val = pixel_format_;

    return true;
}

bool CameraGalaxy::getMinExposure(float& val)
{
    double dValue = 0;
    if (GXGetFloat(hDevice_, GX_FLOAT_CURRENT_ACQUISITION_FRAME_RATE, &dValue) != GX_STATUS_SUCCESS)
    {
        //std::cout << "__________getMinExposure failed___________" << std::endl;
        return false;
    }
    val = (int)(1000000 / dValue);

    return true;
}

bool CameraGalaxy::openCamera()
{
    std::lock_guard<std::mutex> my_guard(operate_mutex_);

    GX_STATUS status = GX_STATUS_SUCCESS;
    uint32_t nDeviceNum = 0;

    status = GXInitLib();
    if (status != GX_STATUS_SUCCESS)
    {
        return false;
    }

    status = GXUpdateDeviceList(&nDeviceNum, 1000);
    if ((status != GX_STATUS_SUCCESS) || (nDeviceNum <= 0))
    {
        return false;
    }

    char cam_idx[8] = "0";
    if (status == GX_STATUS_SUCCESS && nDeviceNum > 0)
    {
        GX_DEVICE_BASE_INFO* pBaseinfo = new GX_DEVICE_BASE_INFO[nDeviceNum];
        size_t nSize = nDeviceNum * sizeof(GX_DEVICE_BASE_INFO);

        status = GXGetAllDeviceBaseInfo(pBaseinfo, &nSize);
        for (int i = 0; i < nDeviceNum; i++)
        {
            if (GX_DEVICE_CLASS_U3V == pBaseinfo[i].deviceClass)
            {
                snprintf(cam_idx, 8, "%d", i + 1);
            }
        }

        delete[] pBaseinfo;
    }

    GX_OPEN_PARAM stOpenParam;
    stOpenParam.accessMode = GX_ACCESS_EXCLUSIVE;
    stOpenParam.openMode = GX_OPEN_INDEX;
    stOpenParam.pszContent = cam_idx;
    status = GXOpenDevice(&stOpenParam, &hDevice_);

    if (status == GX_STATUS_SUCCESS)
    {
        GX_STATUS status = GX_STATUS_SUCCESS;
        status = GXSetEnum(hDevice_, GX_ENUM_ACQUISITION_MODE, 2);
        if (status != GX_STATUS_SUCCESS)
        {
            std::cout << "Failed to set the continuous collection mode!" << std::endl;

        }

        // 关闭帧率模式  
        status = GXSetEnum(hDevice_, GX_ENUM_ACQUISITION_FRAME_RATE_MODE, GX_ACQUISITION_FRAME_RATE_MODE_OFF);
        if (status != GX_STATUS_SUCCESS)
        {
            std::cout << "Failed to turn off automatic frame rate mode!" << std::endl;

        }

        int64_t nLinkThroughputLimitVal = 40000000;
        status = GXSetInt(hDevice_, GX_INT_DEVICE_LINK_THROUGHPUT_LIMIT,nLinkThroughputLimitVal);
        if (status != GX_STATUS_SUCCESS)
        {
            std::cout << "Failed to set link!" << std::endl;

        }

        status = GXSetFloat(hDevice_, GX_FLOAT_ACQUISITION_FRAME_RATE, 249.0);



        // 先设置一个小曝光，例如1000us，然后查看最大帧率
        double exposure_temp = 1000;
        status = GXSetFloat(hDevice_, GX_FLOAT_EXPOSURE_TIME, exposure_temp);
        double max_frame = 0;
        status = GXGetFloat(hDevice_, GX_FLOAT_CURRENT_ACQUISITION_FRAME_RATE, &max_frame);


        std::cout << "frame:" << max_frame << std::endl;
 
        min_camera_exposure_ = (int)(1000000 / max_frame);



  

        status = GXGetInt(hDevice_, GX_INT_WIDTH, &image_width_);
        status = GXGetInt(hDevice_, GX_INT_HEIGHT, &image_height_);

        // 设置线选择器为LINE1  
        status = GXSetEnum(hDevice_, GX_ENUM_LINE_SELECTOR, GX_ENUM_LINE_SELECTOR_LINE1);
        if (status != GX_STATUS_SUCCESS)
        {
            std::cout << "Failed to set line selector." << std::endl;
            return false;
        }

        // 设置线模式为输出模式  
        status = GXSetEnum(hDevice_, GX_ENUM_LINE_MODE, GX_ENUM_LINE_MODE_OUTPUT);
        if (status != GX_STATUS_SUCCESS)
        {
            std::cout << "Failed to set line mode." << std::endl;
            return false;
        }
        status = GXSetEnum(hDevice_, GX_ENUM_LINE_SOURCE, GX_ENUM_LINE_SOURCE_USEROUTPUT0);
        status = GXSetEnum(hDevice_, GX_ENUM_USER_OUTPUT_SELECTOR, GX_USER_OUTPUT_SELECTOR_OUTPUT0);
        status = GXSetBool(hDevice_, GX_BOOL_USER_OUTPUT_VALUE, false);
        setPixelFormat(8);


        status = GXSetAcqusitionBufferNumber(hDevice_, 72);


        //在打开相机时读取相机的40个参数，生成calib_param.txt参数以及五个bin文件
        //设置使用区域
        status = GXSetEnum(hDevice_, GX_ENUM_USER_DATA_FILED_SELECTOR, GX_USER_DATA_FILED_0);
        if (status != GX_STATUS_SUCCESS) {
            std::cout << "Failed to set the usage zone!" << std::endl;
            std::cout << "Set the use zone return code:" << status << std::endl;

        }
     

        //申请长度为 4K 的 buffer，该 buffer 需要由用户填充数据
        size_t nLength = 4096;
        uint8_t* pSetBuffer = new uint8_t[nLength];




        //获取用户区内容
        status = GXGetBuffer(hDevice_, GX_BUFFER_USER_DATA_FILED_VALUE, pSetBuffer, &nLength);
        if (status != GX_STATUS_SUCCESS) {
            LOG(INFO) << "calib param.txt file does not exist, directly generate lookup table!";
        }
        
        float output_param[40] = {};
        memcpy(output_param, pSetBuffer, sizeof(float) * 40);
       /* for (int i = 0; i < 40; i++) {
            std::cout << output_param[i] << std::endl;

        }*/
 
        //判断本地40个参数是否与相机内一致，不一致则重新生成
        float param[40] = {};
        std::ifstream ifile;
        ifile.open("calib_param.txt");
        if (!ifile.is_open())

        {

            LOG(INFO) << "calib param.txt file does not exist, directly generate lookup table!";
            LOG(INFO) << "Lookup tables are being generated";
            LOG(INFO) << "Wait 30 seconds........";
            set_calib_looktable(output_param, image_width_, image_height_);
            LOG(INFO) << "The lookup table generation is complete!";
        }
        else
        {
            
            int j = 0;
            for (int i = 0; i < 40; i++) {
                ifile >> param[i];
                if (output_param[i] == param[i]) {
                    j++;
                }
            }
            std::ifstream file1("combine_xL_rotate_x_cam1_iter.bin");
            std::ifstream file2("combine_xL_rotate_y_cam1_iter.bin");
            std::ifstream file3("calib_param.txt");
            std::ifstream file4("R1.bin");
            std::ifstream file5("single_pattern_mapping.bin");
            std::ifstream file6("single_pattern_minimapping.bin");
            if (file1 && file2 && file3 && file4 && file5 && file6 && (j == 40)) {
             
                LOG(INFO) << "Bin files already exists,No need to generate lookup tables!";
            }
            else
            {
                LOG(INFO) << "The internal parameters are different, directly generate lookup table!";
                LOG(INFO) << "Lookup tables are being generated";
                LOG(INFO) << "Wait 30 seconds........";
                set_calib_looktable(output_param, image_width_, image_height_);
                LOG(INFO) << "The lookup table generation is complete!";
            }
        }


    }
    return true;
}


bool CameraGalaxy::closeCamera()
{
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::lock_guard<std::mutex> my_guard(operate_mutex_);

    if (!camera_opened_state_)
    {
        return false;
    }

    GX_STATUS status = GX_STATUS_SUCCESS;
    status = GXCloseDevice(hDevice_);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "关闭设备失败！" << std::endl;

    }
    status = GXCloseLib();
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "关闭设备库失败！" << std::endl;

    }
    camera_opened_state_ = false;
    return true;
}
bool CameraGalaxy::switchToInternalTriggerMode()
{
    std::lock_guard<std::mutex> my_guard(operate_mutex_);
    GX_STATUS status;
    status = GXSetEnum(hDevice_, GX_ENUM_TRIGGER_MODE, GX_TRIGGER_MODE_ON);
    if (GX_STATUS_SUCCESS != status)
    {
        return false;
    }

    status = GXSetEnum(hDevice_, GX_ENUM_TRIGGER_SOURCE, GX_TRIGGER_SOURCE_SOFTWARE);
    if (GX_STATUS_SUCCESS != status)
    {
        return false;
    }

    status = GXSetEnum(hDevice_, GX_ENUM_EVENT_NOTIFICATION, GX_ENUM_EVENT_NOTIFICATION_ON);
    if (GX_STATUS_SUCCESS != status)
    {
        return false;
    }

    trigger_on_flag_ = false;

    return true;
}
bool CameraGalaxy::switchToExternalTriggerMode()
{

    std::lock_guard<std::mutex> my_guard(operate_mutex_);

    // 设置触发模式为开启  
    GX_STATUS status = GX_STATUS_SUCCESS;
    int trigger_mode = 1;
    status = GXSetEnum(hDevice_, GX_ENUM_TRIGGER_MODE, trigger_mode);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "设置触发模式失败！" << std::endl;

    }

    // 设置触发源为LINE0  
    int trigger_source = 1;
    status = GXSetEnum(hDevice_, GX_ENUM_TRIGGER_SOURCE, trigger_source);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "设置触发源失败！" << std::endl;

    }

    // 设置触发选择器为FRAME_START  
    int trigger_selector = 1;
    status = GXSetEnum(hDevice_, GX_ENUM_TRIGGER_SELECTOR, trigger_selector);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "设置触发选择器失败！" << std::endl;

    }

    // 设置触发激活方式为下降沿触发  
    int trigger_activation = 0;
    status = GXSetEnum(hDevice_, GX_ENUM_TRIGGER_ACTIVATION, trigger_activation);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "设置触发激活方式失败！" << std::endl;

    }

    // 设置用户输出选择器为OUTPUT1  
    status = GXSetEnum(hDevice_, GX_ENUM_USER_OUTPUT_SELECTOR, GX_USER_OUTPUT_SELECTOR_OUTPUT1);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "设置用户输出选择器失败！" << std::endl;

    }

    // 设置用户输出值为真  
    bool user_output_value = true;
    status = GXSetBool(hDevice_, GX_BOOL_USER_OUTPUT_VALUE, user_output_value);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "设置用户输出值失败！" << std::endl;
    }
    trigger_on_flag_ = true;
    return true;
}

bool CameraGalaxy::getExposure(double& val)
{
    GX_STATUS status = GX_STATUS_SUCCESS;
    status = GXGetFloat(hDevice_, GX_FLOAT_EXPOSURE_TIME, &val);

    if (status != GX_STATUS_SUCCESS)
    {
        return false;
    }

    return true;
}
bool CameraGalaxy::setExposure(double val)
{

  /*  if (trigger_on_flag_)
    {
        if (val < min_camera_exposure_)
        {
            val = min_camera_exposure_;
        }
    }
    if (val > max_camera_exposure_)
    {
        val = max_camera_exposure_;
    }*/
    //val = 10000;
    //std::cout << "exposureeee: " << val << std::endl;
    GX_STATUS status = GX_STATUS_SUCCESS;
    status = GXSetFloat(hDevice_, GX_FLOAT_EXPOSURE_TIME, val);

    if (status != GX_STATUS_SUCCESS)
    {
        return false;
    }

    return true;
}
bool CameraGalaxy::getGain(double& val)
{
    std::lock_guard<std::mutex> my_guard(operate_mutex_);

    GX_STATUS status = GX_STATUS_SUCCESS;
    status = GXGetFloat(hDevice_, GX_FLOAT_GAIN, &val);

    if (status != GX_STATUS_SUCCESS)
    {

        return false;
    }

    return true;
}
bool CameraGalaxy::setGain(double val)
{
    std::lock_guard<std::mutex> my_guard(operate_mutex_);

    GX_STATUS status = GX_STATUS_SUCCESS;
    status = GXSetFloat(hDevice_, GX_FLOAT_GAIN, val);

    if (status != GX_STATUS_SUCCESS)
    {
        return false;
    }
    return true;
}


void CameraGalaxy::LINE0_IN() {
    GX_STATUS status = GX_STATUS_SUCCESS;

    // 设置线选择器为LINE0  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_SELECTOR, GX_ENUM_LINE_SELECTOR_LINE0);
    if (status != GX_STATUS_SUCCESS)
    {
        printf("Failed to set line selector.\n");
        return;
    }
    // 设置线模式为输入模式  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_MODE, GX_ENUM_LINE_MODE_INPUT);
    if (status != GX_STATUS_SUCCESS)
    {
        printf("Failed to set line mode.\n");
        return;
    }
}

void CameraGalaxy::LINE1_OUT(bool xOn) {
    GX_STATUS status = GX_STATUS_SUCCESS;

    // 设置线选择器为LINE1  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_SELECTOR, GX_ENUM_LINE_SELECTOR_LINE1);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line selector." << std::endl;
        return;
    }

    // 设置线反转  
    //status = GXSetBool(hDevice_, GX_BOOL_LINE_INVERTER, !xOn);
    //if (status != GX_STATUS_SUCCESS)
    //{
    //    std::cout << "Failed to set line inverter." << std::endl;
    //    return;
    //}
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_SOURCE, GX_ENUM_LINE_SOURCE_USEROUTPUT0);
    status = GXSetEnum(hDevice_, GX_ENUM_USER_OUTPUT_SELECTOR, GX_USER_OUTPUT_SELECTOR_OUTPUT0);
    status = GXSetBool(hDevice_, GX_BOOL_USER_OUTPUT_VALUE, xOn);
}

void CameraGalaxy::SDA_OUTPUT() {
    GX_STATUS status = GX_STATUS_SUCCESS;

    // 设置线选择器为LINE2  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_SELECTOR, GX_ENUM_LINE_SELECTOR_LINE2);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line selector." << std::endl;
        return;
    }
    // 设置线模式为输出模式  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_MODE, GX_ENUM_LINE_MODE_OUTPUT);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line mode." << std::endl;
        return;
    }
    // 设置线反转为关闭  
    status = GXSetBool(hDevice_, GX_BOOL_LINE_INVERTER, false);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line inverter." << std::endl;
        return;
    }
}

void CameraGalaxy::SDA_ON() {
    GX_STATUS status = GX_STATUS_SUCCESS;

    // 设置线选择器为LINE2  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_SELECTOR, GX_ENUM_LINE_SELECTOR_LINE2);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line selector." << std::endl;
        return;
    }
    // 设置线模式为输出模式  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_MODE, GX_ENUM_LINE_MODE_OUTPUT);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line mode." << std::endl;
        return;
    }
    // 设置线反转为关闭  
    status = GXSetBool(hDevice_, GX_BOOL_LINE_INVERTER, false);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line inverter." << std::endl;
        return;
    }
}

void CameraGalaxy::SDA_OFF() {
    GX_STATUS status = GX_STATUS_SUCCESS;

    // 设置线选择器为LINE2  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_SELECTOR, GX_ENUM_LINE_SELECTOR_LINE2);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line selector." << std::endl;
        return;
    }
    // 设置线模式为输出模式  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_MODE, GX_ENUM_LINE_MODE_OUTPUT);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line mode." << std::endl;
        return;
    }
    // 设置线反转为关闭  
    status = GXSetBool(hDevice_, GX_BOOL_LINE_INVERTER, true);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line inverter." << std::endl;
        return;
    }
}

void CameraGalaxy::SDA_INPUT() {
    GX_STATUS status = GX_STATUS_SUCCESS;

    // 设置线选择器为LINE2  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_SELECTOR, GX_ENUM_LINE_SELECTOR_LINE2);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line selector2." << std::endl;
        return;
    }
    // 设置线模式为输出模式  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_MODE, GX_ENUM_LINE_MODE_INPUT);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line mode." << std::endl;
        return;
    }
    // 设置线反转为关闭  
    status = GXSetBool(hDevice_, GX_BOOL_LINE_INVERTER, false);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line inverter." << std::endl;
        return;
    }
}

bool CameraGalaxy::SDA_READ() {
    GX_STATUS status = GX_STATUS_SUCCESS;
    bool bLineStatus = true;
    // 设置线选择器为LINE2  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_SELECTOR, GX_ENUM_LINE_SELECTOR_LINE2);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line selector22." << std::endl;
        return 0;
    }
    // 设置线模式为输出模式  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_MODE, GX_ENUM_LINE_MODE_INPUT);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line mode." << std::endl;
        return 0;
    }
    status = GXGetBool(hDevice_, GX_BOOL_LINE_STATUS, &bLineStatus);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to get line status." << std::endl;
        return 0;
    }
    return bLineStatus;
}

void CameraGalaxy::SCL_OUTPUT() {

    GX_STATUS status = GX_STATUS_SUCCESS;

    // 设置线选择器为LINE2  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_SELECTOR, GX_ENUM_LINE_SELECTOR_LINE3);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line selector3." << std::endl;
        return;
    }
    // 设置线模式为输出模式  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_MODE, GX_ENUM_LINE_MODE_OUTPUT);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line mode." << std::endl;
        return;
    }
    // 设置线反转为关闭  
    status = GXSetBool(hDevice_, GX_BOOL_LINE_INVERTER, false);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line inverter." << std::endl;
        return;
    }

}
void CameraGalaxy::SCL_ON_OFF() {
    GX_STATUS status = GX_STATUS_SUCCESS;

    // 设置线选择器为LINE2  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_SELECTOR, GX_ENUM_LINE_SELECTOR_LINE3);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line selector33." << std::endl;
        return;
    }
    // 设置线反转为关闭  
    status = GXSetBool(hDevice_, GX_BOOL_LINE_INVERTER, false);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line inverter." << std::endl;
        return;
    }
    // 设置线反转为关闭  
    status = GXSetBool(hDevice_, GX_BOOL_LINE_INVERTER, true);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line inverter." << std::endl;
        return;
    }
}
void CameraGalaxy::SCL_ON() {

    GX_STATUS status = GX_STATUS_SUCCESS;

    // 设置线选择器为LINE2  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_SELECTOR, GX_ENUM_LINE_SELECTOR_LINE3);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line selector33." << std::endl;
        return;
    }
    // 设置线模式为输出模式  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_MODE, GX_ENUM_LINE_MODE_OUTPUT);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line mode." << std::endl;
        return;
    }
    // 设置线反转为关闭  
    status = GXSetBool(hDevice_, GX_BOOL_LINE_INVERTER, false);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line inverter." << std::endl;
        return;
    }

}

void CameraGalaxy::SCL_OFF() {
    GX_STATUS status = GX_STATUS_SUCCESS;

    // 设置线选择器为LINE2  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_SELECTOR, GX_ENUM_LINE_SELECTOR_LINE3);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line selector333." << std::endl;
        return;
    }
    // 设置线模式为输出模式  
    status = GXSetEnum(hDevice_, GX_ENUM_LINE_MODE, GX_ENUM_LINE_MODE_OUTPUT);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line mode." << std::endl;
        return;
    }
    // 设置线反转为关闭  
    status = GXSetBool(hDevice_, GX_BOOL_LINE_INVERTER, true);
    if (status != GX_STATUS_SUCCESS)
    {
        std::cout << "Failed to set line inverter." << std::endl;
        return;
    }
}

size_t CameraGalaxy::read(char inner_reg, void* buffer, size_t buffer_size) {
    return 0;
    char ack = 0;
    char inner_addr = 0x1b;

    SDA_OUTPUT();
    SDA_ON();
    SCL_ON();

    SDA_OFF();
    SCL_OFF();

    char ch = (inner_addr << 1);

    for (int j = 0; j < 8; j++)
    {
        if (ch & 0x80)
        {
            SDA_ON();
        }
        else
        {
            SDA_OFF();
        }
        SCL_ON();
        SCL_OFF();
        ch <<= 1;
    }
    SDA_ON();
    SDA_INPUT();
    SCL_ON();
    ack = SDA_READ();
    SCL_OFF();
    if (ack)
    {
        std::cout << "no ack" << std::endl;
    }
    SDA_OUTPUT();

    ch = inner_reg;
    SDA_OUTPUT();
    for (int j = 0; j < 8; j++)
    {
        if (ch & 0x80)
        {
            SDA_ON();
        }
        else
        {
            SDA_OFF();
        }
        SCL_ON();
        SCL_OFF();
        ch <<= 1;
    }
    SDA_ON();
    SDA_INPUT();
    SCL_ON();
    ack = SDA_READ();
    SCL_OFF();
    if (ack)
    {
        std::cout << "no ack" << std::endl;
    }

    SDA_OUTPUT();
    SCL_OFF();
    SDA_OFF();

    SCL_ON();
    SDA_ON();

    SDA_OFF();
    SCL_OFF();


    ch = (inner_addr << 1) | 0x01;

    for (int j = 0; j < 8; j++)
    {
        if (ch & 0x80)
        {
            SDA_ON();
        }
        else
        {
            SDA_OFF();
        }
        SCL_ON();
        SCL_OFF();
        ch <<= 1;
    }
    SDA_ON();
    SDA_INPUT();
    SCL_ON();
    ack = SDA_READ();
    SCL_OFF();
    if (ack)
    {
        std::cout << "no ack" << std::endl;
    }
    char* p = (char*)buffer;
    for (int i = 0; i < buffer_size; i++)
    {
        ch = 0;
        SDA_INPUT();
        for (int j = 0; j < 8; j++)
        {
            ch <<= 1;
            if (SDA_READ())
            {
                ch |= 0x01;
                //  std::cout << "1 ";
            }
            else
            {
                //  std::cout << "0 ";
            }
            SCL_ON();
            SCL_OFF();
        }
        std::cout << std::endl;
        SDA_OUTPUT();
        SCL_ON();
        SDA_OFF();
        SCL_OFF();
        *p++ = (ch & 0xff);
        std::cout << "rd:" << (int)ch << std::endl;
    }

    SCL_ON();
    SDA_ON();

    return 0;
}

size_t CameraGalaxy::write(char inner_reg, void* buffer, size_t buffer_size) {
    char ack = 0;
    char inner_addr = 0x1b;
    SDA_OUTPUT();
    SDA_ON();
    SCL_ON();

    SDA_OFF();
    SCL_OFF();

    char ch = (inner_addr << 1);

    for (int j = 0; j < 8; j++)
    {
        if (ch & 0x80)
        {
            SDA_ON();
        }
        else
        {
            SDA_OFF();
        }
        /*SCL_ON();
        SCL_OFF();*/
        SCL_ON_OFF();
        ch <<= 1;
    }
    SDA_ON();
    SDA_INPUT();
    SCL_ON();
    ack = SDA_READ();
    SCL_OFF();
    if (ack)
    {
        std::cout << "no ack" << std::endl;
    }
    SDA_OUTPUT();

    //新加inner_reg;
    ch = inner_reg;
    for (int j = 0; j < 8; j++) {
        if (ch & 0x80) {
            SDA_ON();
        }
        else
        {
            SDA_OFF();
        }
        /*SCL_ON();
        SCL_OFF();*/
        SCL_ON_OFF();
        ch <<= 1;
    }
    SDA_ON();
    SDA_INPUT();
    SCL_ON();
    ack = SDA_READ();
    SCL_OFF();
    if (ack) {
        std::cout << "no ack" << std::endl;
    }
    SDA_OUTPUT();


    char* p = (char*)buffer;
    for (int i = 0; i < buffer_size; i++)
    {
        char ch = *p++;
        SDA_OUTPUT();
        for (int j = 0; j < 8; j++)
        {
            if (ch & 0x80)
            {
                SDA_ON();
            }
            else
            {
                SDA_OFF();
            }
            /*SCL_ON();
            SCL_OFF();*/
            SCL_ON_OFF();
            ch <<= 1;
        }
        SDA_ON();
        SDA_INPUT();
        SCL_ON();
        ack = SDA_READ();
        SCL_OFF();
        if (ack)
        {
            std::cout << "no ack" << std::endl;
        }

    }
    SDA_OUTPUT();
    SCL_OFF();
    SDA_OFF();

    SCL_ON();
    SDA_ON();
    return 0;
}

void CameraGalaxy::trigger_one_sequence()
{

    EnterCriticalSection();
    LINE1_OUT(true);
    LINE1_OUT(false);
    LeaveCriticalSection();
}

void CameraGalaxy::trigger_hdr_list_flush()
{
    EnterCriticalSection();
    LINE1_OUT(true);
    Sleep(100);
    LINE1_OUT(false);
    LeaveCriticalSection();
}

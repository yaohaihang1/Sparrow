#include "lightcrafter3010.h"
#include <stdio.h>
#include <string.h>
#include "easylogging++.h"
#include "math.h"
#include "protocol.h"

LightCrafter3010::LightCrafter3010()
{
    camera_exposure_ = 12000;
    camera_exposure_setted_ = 0;
    dlp_min_exposure_ = 1700;
    camera_min_exposure_ = 6000;
    is_hdr_mode_ = false;
    pattern_mode_now_ = -1;
    hdr_nums_ = 0;
}
 
size_t LightCrafter3010::read_with_param(char inner_addr,unsigned char param, void* buffer, size_t buffer_size)
{
    return	0;
}

LightCrafter3010::~LightCrafter3010()
{

}


int LightCrafter3010::init()
{   
  
    unsigned char working_mode = 0; // 初始化为单曝光模式
    write(SingleChip_Write_Working_Mode, &working_mode, 1);
    is_hdr_mode_ = false;

    int min_exposure_int = camera_min_exposure_; // 初始化相机的最小曝光时间
    write(SingleChip_Write_Min_Exposure, &min_exposure_int, 4);
    return DF_SUCCESS;
}


int LightCrafter3010::read_patterns_sets_num(int &num)
{
    num = 9;

    return 0;
}
  
void LightCrafter3010::set_trigger_out_delay(int delay_time)
{
}

int LightCrafter3010::SetLedCurrent(unsigned short R, unsigned short G, unsigned short B)
{
    int bright = B;
    write(SingleChip_Write_Led_Current, &bright, 4);

    return DF_SUCCESS;
} 


void LightCrafter3010::read_dmd_device_id(int& version)
{
    version = 3010;
}

void LightCrafter3010::set_camera_min_exposure(float min)
{
    if ((int)camera_min_exposure_ != (int)min)
    {
        //std::cout << "_______________different min exposure______________" << min << std::endl;
        camera_min_exposure_ = (int)min;
        int min_exposure_int = min;
        write(SingleChip_Write_Min_Exposure, &min_exposure_int, 4);
        return;
    }
    //std::cout << "_______________same min exposure______________" << min << std::endl;
}

void LightCrafter3010::enable_solid_field()
{
    write(SingleChip_Write_Enable_Solid_Field, NULL, 0);
    Sleep(100);
}

void LightCrafter3010::disable_solid_field()
{
    write(SingleChip_Write_Disable_Checkerboard, NULL, 0);
}

void LightCrafter3010::enable_checkerboard()
{
    write(SingleChip_Write_Enable_Checkerboard, NULL, 0);
} 

void LightCrafter3010::disable_checkerboard()
{
    write(SingleChip_Write_Disable_Checkerboard, NULL, 0);
}

void LightCrafter3010::set_internal_pattern_stop()
{
    std::cout << "待实现：set_internal_pattern_stop()" << std::endl;
}

void LightCrafter3010::set_flash_data_type()
{
    std::cout << "待实现：set_flash_data_type()" << std::endl;
}

bool LightCrafter3010::set_flash_build_data_size(unsigned int data_size)
{
    std::cout << "待实现：set_flash_build_data_size()" << std::endl;
    return true;
}

void LightCrafter3010::set_erase_flash()
{
    std::cout << "待实现：set_erase_flash()" << std::endl;
}

bool LightCrafter3010::check_erase_flash_status()
{
    return true;
}

void LightCrafter3010::set_flash_data_length(unsigned short dataLen)
{
}

int LightCrafter3010::write_data_into_the_flash(unsigned char writeFlashCmd, char *TxBuffer, unsigned short dataLen)
{
}

void LightCrafter3010::read_data_from_the_flash(unsigned char readFlashCmd, char *RxBuffer, unsigned short dataLen)
{
}

void LightCrafter3010::reload_pattern_order_table_from_flash()
{
}

int LightCrafter3010::write_pattern_table(unsigned char* pattern_index, unsigned char* pattern_nums,unsigned char* pattern_invert, int len,float camera_exposure)
{
    return DF_SUCCESS;
}


int LightCrafter3010::write_pattern_table(unsigned char* pattern_index, unsigned char* pattern_nums, int len,float camera_exposure)
{
    return DF_SUCCESS;
}

void LightCrafter3010::write_pattern_table(unsigned char* pattern_index, int len)
{
}

void LightCrafter3010::set_pattern_mode(int i)
{
    if (i != pattern_mode_now_)
    {
        unsigned char mode = i;
        write(SingleChip_Write_Pattern_Mode, &mode, 1);
        pattern_mode_now_ = i;
    }
}

void LightCrafter3010::pattern_mode01()
{
    set_pattern_mode(1);
}

void LightCrafter3010::pattern_mode02()
{
    set_pattern_mode(2);
}

void LightCrafter3010::pattern_mode_brightness()
{
    std::cout << "待实现：pattern_mode_brightness()" << std::endl;
}

void LightCrafter3010::pattern_mode03()
{
    set_pattern_mode(3);
}


int LightCrafter3010::pattern_mode04_repetition(int repetition_count)
{
    std::cout << "待实现：pattern_mode04_repetition()" << std::endl;
    return DF_SUCCESS;
}

void LightCrafter3010::pattern_mode03_repetition(int repetition_count)
{
    std::cout << "待实现：pattern_mode03_repetition()" << std::endl;
}

int LightCrafter3010::pattern_mode04()
{
    set_pattern_mode(4);

    return DF_SUCCESS;
}

int LightCrafter3010::pattern_mode05()
{
    set_pattern_mode(5);

    return DF_SUCCESS;
}

int LightCrafter3010::pattern_mode06()
{
    set_pattern_mode(6);

    return DF_SUCCESS;
}

int LightCrafter3010::pattern_mode08() 
{
    set_pattern_mode(8);

    return DF_SUCCESS;
}

void LightCrafter3010::read_pattern_status()
{
}

void LightCrafter3010::read_pattern_table(int i)
{
}


void LightCrafter3010::start_pattern_sequence()
{
    trigger_one_sequence();
	//write(SingleChip_Write_Start_Pattern_Sequence, NULL, 0);
}


void LightCrafter3010::stop_pattern_sequence()
{
    std::cout << "待实现：stop_pattern_sequence()" << std::endl;
}

float LightCrafter3010::get_temperature()
{
    std::cout << "待实现：get_temperature()" << std::endl;
    return 0;
}

size_t LightCrafter3010::read_mcp3221(void* buffer, size_t buffer_size)
{
    return 0;
}

float LightCrafter3010::lookup_table(float fRntc)
{
    float temperature = 30.0;
    float last = 0, current = 0;
    int i = 0, number = 0;

    last = abs(fRntc - R_table[0]);

    for (i = 0; i < R_TABLE_NUM; i++)
    {
        current = abs(fRntc - R_table[i]);

        if (current < last) 
        {
            last = current;
            number = i;
        }
    }

    temperature = number - 40.0;

    printf("temperature = %f, R_table[%d] = %f\n", temperature, number, R_table[number]);

    return temperature;
}

float LightCrafter3010::get_projector_temperature()
{
    unsigned char buffer[2];
    int size = read_mcp3221(buffer, 2);

    short OutputCode;
    float temperature;
    if (size == 2) 
    {
        OutputCode = ((buffer[0] << 8) & 0xff00) | buffer[1];
        printf("The AD data = 0x%x = %d\n", OutputCode, OutputCode);

        // Rntc = 10 * (4096 - AD) / AD, unit=KO
        float fAD = OutputCode;
        float fRntc = 10.0 * (4096.0 - fAD) / fAD;
        printf("R = %f\n", fRntc);

        temperature = lookup_table(fRntc);
        temperature = (temperature >= 125.0) ? -125.0 : temperature;
    } 
    else
    {
        temperature = -100;
    }

    return temperature;
}

void LightCrafter3010::set_camera_exposure(float exposure)
{
    if ((int)exposure != camera_exposure_setted_)
    {
        camera_exposure_ = (int)exposure;
        camera_exposure_setted_ = (int)exposure;
        int exposure_int = camera_exposure_;
        write(SingleChip_Write_Exposure_Time, &exposure_int, 4);
    }
}

void LightCrafter3010::enable_hdr_mode()
{
    trigger_hdr_list_flush();
    if (!is_hdr_mode_)
    {
        is_hdr_mode_ = true;
        unsigned char working_mode = 1;
        write(SingleChip_Write_Working_Mode, &working_mode, 1);
    }
}

void LightCrafter3010::disable_hdr_mode()
{
    std::cout << "____________close HDR mode___________" << std::endl;
    if (is_hdr_mode_)
    {
        is_hdr_mode_ = false;
        std::cout << "____________close HDR mode______success_____" << std::endl;
        unsigned char working_mode = 0;
        write(SingleChip_Write_Working_Mode, &working_mode, 1);
    }
}

bool compare_hdr_list(int hdr_nums, std::vector<int> hdr_exposure_list, std::vector<int> hdr_current_list, int hdr_nums_new, std::vector<int> hdr_exposure_list_new, std::vector<int> hdr_current_list_new)
{
    if (hdr_nums != hdr_nums_new || hdr_exposure_list.size() < hdr_nums || hdr_exposure_list_new.size() < hdr_nums || hdr_current_list.size() < hdr_nums || hdr_current_list_new.size() < hdr_nums)
    {
        std::cout << "HDR list is different1!" << std::endl;
        return false;
    }

    for (int i = 0; i < hdr_nums; i += 1)
    {
        if (hdr_exposure_list[i] != hdr_exposure_list_new[i] || hdr_current_list[i] != hdr_current_list_new[i])
        {
            std::cout << "HDR list is different2!" << std::endl;
            return false;
        }
    }

    std::cout << "HDR list is the same!" << std::endl;
    return true;
}

struct HdrPair
{
    int current;
    int exposure;
};

void LightCrafter3010::set_hdr_list(int hdr_nums, std::vector<int> hdr_exposure_list, std::vector<int> hdr_current_list)
{
    // 比较即将设置的数据与旧数据的差异，若存在差异则需要设置
    if (!compare_hdr_list(hdr_nums_, hdr_exposure_list_, hdr_current_list_, hdr_nums, hdr_exposure_list, hdr_current_list))
    {
        hdr_nums_ = hdr_nums;
        hdr_exposure_list_ = hdr_exposure_list;
        hdr_current_list_ = hdr_current_list;

        unsigned char hdr_list[49]; // 4 * 2 * 6 + 1
        hdr_list[0] = hdr_nums&0xff;
        HdrPair* hdr_pair = (HdrPair*)&hdr_list[1];
        for (int i = 0; i < hdr_nums; i += 1)
        {
            std::cout << "hdr_current_list[i]: " << hdr_current_list[i] << std::endl;
            std::cout << "hdr_exposure_list[i]: " << hdr_exposure_list[i] << std::endl;
            std::cout << "---------------------------------" << std::endl;
            hdr_pair->current = hdr_current_list[i];
            hdr_pair->exposure = hdr_exposure_list[i];
            hdr_pair += 1;
        }
        int* test = (int*)&hdr_list[1];

        for (int i = 0; i < hdr_nums; i += 1)
        {
            std::cout << test[0] << std::endl;
            std::cout << test[1] << std::endl;
            std::cout << "________________" << std::endl;
            test += 2;
        }

        write(SingleChip_Write_Hdr_Data, hdr_list, hdr_nums_ * 2 * 4 + 1);
    }

    return;
}


#pragma once
//#include "i2c.h"
#include <stdio.h>
#include <string.h>

#define PWR_CTRL_I2C_CMD_LEN	1

class PowerController
{
private:
	//I2CDevice mcu;
	size_t read(char inner_addr, void* buffer, size_t buffer_size) { return 0; };
	size_t write(char inner_addr, void* buffer, size_t buffer_size) { return 0; };

public:
	PowerController() { };
	~PowerController() { };

    bool off_projector() { return 0; };
    bool off_board() { return 0; };

private:

};


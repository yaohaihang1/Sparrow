#pragma once
//#define WIN32_LEAN_AND_MEAN 
//#include <Windows.h>
#include <winsock2.h>  
//#include<Windows.h>
//#include<ws2ipdef.h>
#include <assert.h>
#include <iostream>

#include <errno.h> 

#pragma comment(lib, "ws2_32.lib")

int setup_socket(int port);

int accept_new_connection(int server_sock); 

int send_buffer(int sock, const char* buffer, int buffer_size);

int recv_buffer(int sock, char* buffer, int buffer_size);

int send_command(int sock, int command);
 
int recv_command(int sock, int* command);
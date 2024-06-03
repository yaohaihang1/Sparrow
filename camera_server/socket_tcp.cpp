#include "socket_tcp.h"
#include "protocol.h"
#include "easylogging++.h"
#include <cstring> 
#include<iostream>
#define INET_ADDRSTRLEN 16
#define _CRT_SECURE_NO_WARNINGS

int setup_socket(int port)
{

    WORD sockVersion = MAKEWORD(2, 2);
    WSADATA data;
    if (WSAStartup(sockVersion, &data) != 0)
    {
        LOG(ERROR) << "WSAStartup error!";
        return DF_FAILED;
    }
    int server_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if(server_sock<0)
    {
        perror("ERROR: socket()");
        exit(0);
    }

    // 启用TCP层保活机制  
    BOOL enable = TRUE;
    setsockopt(server_sock, SOL_SOCKET, SO_KEEPALIVE, (const char*)&enable, sizeof(enable));


    //将套接字和IP、端口绑定
    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));  //每个字节都用0填充
    serv_addr.sin_family = AF_INET;  //使用IPv4地址
    serv_addr.sin_addr.s_addr=INADDR_ANY;  //具体的IP地址
    serv_addr.sin_port = htons(port);  //端口

    int ret = bind(server_sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr));


  /*  if (ret == INVALID_SOCKET) {
        std::cout << "Failed to create socket. Error code: " << WSAGetLastError() << std::endl;
        WSACleanup();
        return DF_FAILED;
    }*/

    if(ret==-1)
    {
        //printf("bind ret=%d, %s\n", ret, strerror(errno));
        closesocket(server_sock);
        return DF_FAILED;
    }

    //进入监听状态，等待用户发起请求
    ret = listen(server_sock, 1);

   
    if(ret == -1)
    {

        //printf("listen ret=%d, %s\n", ret, strerror(errno));
        closesocket(server_sock);
        return DF_FAILED;
    }
    return server_sock;
}


int accept_new_connection(int server_sock)
{
    //std::cout<<"listening"<<std::endl;
    //接收客户端请求
    sockaddr_in clnt_addr;
    int clnt_addr_size = sizeof(clnt_addr);
    int client_sock = accept(server_sock, (struct sockaddr*)&clnt_addr, &clnt_addr_size);


    //windows超时设置
    int time_out = 30 * 1000;
    //发送时限
    setsockopt(server_sock, SOL_SOCKET, SO_SNDTIMEO, (char*)&time_out, sizeof(time_out));
    //接收时限  
    setsockopt(server_sock, SOL_SOCKET, SO_RCVTIMEO, (char*)&time_out, sizeof(time_out));

    LOG(INFO) << "client_sock: " << client_sock;;
    std::cout << "client_sock: " << client_sock << std::endl;
    return client_sock;
}


int send_buffer(int sock, const char* buffer, int buffer_size)
{
    int size = 0;
    int ret = send(sock, (char*)&buffer_size, sizeof(buffer_size), 0);
    LOG(INFO) << "send buffer_size ret=" << ret;
    if (ret == -1)
    {
        return DF_FAILED;
    }

    int sent_size = 0;
    ret = send(sock, buffer, buffer_size, 0);
    LOG(INFO) << "send buffer ret=" << ret;
    if (ret == -1)
    {
        return DF_FAILED;
    }
    sent_size += ret;
    while (sent_size != buffer_size)
    {
        buffer += ret;
        LOG(INFO) << "sent_size=" << sent_size;
        ret = send(sock, buffer, buffer_size - sent_size, 0);
        LOG(INFO) << "ret=" << ret;
        if (ret == -1)
        {
            return DF_FAILED;
        }
        sent_size += ret;
    }

    return DF_SUCCESS;
}

int recv_buffer(int sock, char* buffer, int buffer_size)
{
    int size = 0;
    int ret = recv(sock, (char*)&size, sizeof(size), 0);
    if (buffer_size < size)
    {
        LOG(ERROR) << "buffer_size < size";
        LOG(INFO) << "buffer_size= " << buffer_size;
        LOG(INFO) << "size= " << size;
        return DF_FAILED;
    }
    int n_recv = 0;
    int null_flag = 0;

    while (ret != -1)
    {
        ret = recv(sock, buffer, buffer_size, 0);
        
        LOG(INFO) << "recv: " << "ret=" << ret << std::endl;
        if (ret > 0)
        {
            buffer_size -= ret;
            n_recv += ret;
            buffer += ret;
        }
        else if (0 == ret)
        {
            null_flag++;
        }

        if (null_flag > 100)
        {
            LOG(INFO) << "recv_buffer failed!";
            return DF_FAILED;
        }

        if (buffer_size == 0)
        {
            //assert(n_recv == size);
            if (n_recv != size)
            {
                LOG(ERROR) << "recv err: n_recv != size ";
                return DF_FAILED;
            }

            return DF_SUCCESS;
        }
    }
    return DF_FAILED;
}

int send_command(int sock, int command)
{
    return send_buffer(sock, (const char*)&command, sizeof(int));
}

int recv_command(int sock, int* command)
{
    return recv_buffer(sock, (char*)command, sizeof(int));
}
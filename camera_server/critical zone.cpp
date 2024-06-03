#include"critical zone.h"

// 全局变量声明
CRITICAL_SECTION g_cs;

// 初始化临界区
void InitializeCriticalSection() {
    InitializeCriticalSection(&g_cs);
}

// 清理临界区资源
void DeleteCriticalSection() {
    DeleteCriticalSection(&g_cs);
}


// 进入临界区
void EnterCriticalSection() {

    EnterCriticalSection(&g_cs);
}


// 离开临界区
void LeaveCriticalSection() {
    
    LeaveCriticalSection(&g_cs);
}
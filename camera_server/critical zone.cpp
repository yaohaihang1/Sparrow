#include"critical zone.h"

// ȫ�ֱ�������
CRITICAL_SECTION g_cs;

// ��ʼ���ٽ���
void InitializeCriticalSection() {
    InitializeCriticalSection(&g_cs);
}

// �����ٽ�����Դ
void DeleteCriticalSection() {
    DeleteCriticalSection(&g_cs);
}


// �����ٽ���
void EnterCriticalSection() {

    EnterCriticalSection(&g_cs);
}


// �뿪�ٽ���
void LeaveCriticalSection() {
    
    LeaveCriticalSection(&g_cs);
}
#pragma once
#include <windows.h>
#include <iostream>


// ��ʼ���ٽ���
void InitializeCriticalSection();

// �����ٽ�����Դ
void DeleteCriticalSection();


// �����ٽ���
void EnterCriticalSection();

//�뿪�ٽ���
void LeaveCriticalSection();


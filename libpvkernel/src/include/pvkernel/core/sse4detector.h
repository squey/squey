#ifndef PVCORE_SSE4DETECTOR_H
#define PVCORE_SSE4DETECTOR_H

#include <pvkernel/core/stdint.h>

struct CPUIDinfo
{ 
	uint32_t EAX,EBX,ECX,EDX; 
};

int isSSE41Supported(void);
int isSSE41andSSE42supported(void);
int isGenuineIntel(void);
int isCPUIDsupported(void);


#endif

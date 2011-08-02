#include <iostream>

#include <stdlib.h>

#include <pvcore/sysinfo.h>

int main(void)
{
#include "test-env.h"

	PVCore::SysInfo sysinfo;

	std::cout << sysinfo.gpu_count();

	return 0;
}

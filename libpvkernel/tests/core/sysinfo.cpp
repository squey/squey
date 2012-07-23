/**
 * \file sysinfo.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <iostream>

#include <stdlib.h>

#include <pvkernel/core/sysinfo.h>

int main(void)
{
#include "test-env.h"

	PVCore::SysInfo sysinfo;

	std::cout << sysinfo.gpu_count();

	return 0;
}

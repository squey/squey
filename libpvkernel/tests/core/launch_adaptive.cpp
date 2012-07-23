/**
 * \file launch_adaptive.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVParallels.h>
#include <iostream>

void f1()
{
	std::cout << "start f1, sleep 5s" << std::endl;
	for (int i = 0; i < 5; i++) {
		sleep(1);
		boost::this_thread::interruption_point();
		std::cout << "hi" << std::endl;
	}
	std::cout << "end f1" << std::endl;
}

void f2()
{
	std::cout << "in f2" << std::endl;
}

int main()
{
	PVCore::launch_adaptive(f1, f2, boost::posix_time::seconds(1));
	return 0;
}

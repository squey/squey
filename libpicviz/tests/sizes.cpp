/**
 * \file sizes.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <picviz/PVIndexArray.h>
#include <picviz/PVView.h>

#include <iostream>

int main()
{
	std::cout << "sizeof(PVIndexArray) = " << sizeof(Picviz::PVIndexArray) << std::endl;
	std::cout << "sizeof(PVView) = " << sizeof(Picviz::PVView) << std::endl;

	return 0;
}

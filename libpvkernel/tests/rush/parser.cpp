/**
 * \file parser.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/rush/PVXmlParamParser.h>
#include <iostream>

int main(int argc, char** argv)
{
	if (argc < 1) {
		std::cerr << "Usage: " << argv[0] << "file_format" << std::endl;
		return 1;
	}
	PVRush::PVXmlParamParser parser(argv[1]);

	return 0;
}

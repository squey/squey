/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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

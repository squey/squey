/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include "import_export.h"

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " file format" << std::endl;
		return 1;
	}

	static const std::string input_file = argv[1];
	static const std::string format = argv[2];
	static const std::string ref_file = (argc >= 4) ? argv[3] : input_file;

	std::string output_file = import_export(input_file, format, true /* cancel */);

	std::remove(output_file.c_str());

	return 0;
}

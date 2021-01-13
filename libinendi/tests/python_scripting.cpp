/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2020
 */
#include "import_export.h"

int main(int argc, char** argv)
{
    if (argc <= 3) {
		std::cerr << "Usage: " << argv[0] << " file format ref_file" << std::endl;
		return 1;
	}

	static const std::string input_file = argv[1];
	static const std::string format = argv[2];
	static const std::string ref_file = argv[3];

	std::string output_file = import_export(input_file, format, true);

    bool same_content = PVRush::PVUtils::files_have_same_content(ref_file, output_file);
	std::cout << std::endl << ref_file << " - " << output_file << std::endl;
	PV_ASSERT_VALID(same_content);
	std::remove(output_file.c_str());
}
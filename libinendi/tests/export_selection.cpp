/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "import_export.h"

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " file format [ref_file]" << std::endl;
		return 1;
	}

	static const std::string input_file = argv[1];
	static const std::string format = argv[2];
	static const std::string ref_file = (argc >= 4) ? argv[3] : input_file;

	std::string output_file = import_export(input_file, format);

#ifndef INSPECTOR_BENCH
	std::string file_extension = input_file.substr(input_file.rfind('.') + 1);
	std::string cmd = PVCore::PVStreamingDecompressor::executable(file_extension);
	std::string uncompressed_file = output_file;
	if (not cmd.empty()) {
		std::string output_file2 = import_export(output_file, format);
		uncompressed_file = output_file2.substr(0, output_file2.find_last_of("."));
		if (cmd == "funzip") {
			cmd = "unzip -o -qq";
			uncompressed_file = "-";
		}
		cmd += " " + output_file2;
		int result = system(cmd.c_str());
		PV_VALID(result, 0);
		std::remove(output_file2.c_str());
	}
	bool same_content = PVRush::PVUtils::files_have_same_content(ref_file, uncompressed_file);
	std::cout << std::endl << ref_file << " - " << uncompressed_file << std::endl;
	PV_ASSERT_VALID(same_content);
	std::remove(uncompressed_file.c_str());
#endif // INSPECTOR_BENCH

	std::remove(output_file.c_str());
}

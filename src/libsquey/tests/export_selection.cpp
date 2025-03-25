//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include "import_export.h"
#include <boost/algorithm/string/join.hpp>
#include <boost/filesystem.hpp>

int main(int argc, char** argv)
{
	if (argc <= 3) {
		std::cerr << "Usage: " << argv[0] << " file format test_selection [ref_file]" << std::endl;
		return 1;
	}

	static const std::string input_file = argv[1];
	static const std::string format = argv[2];
	static const bool test_selection = std::string(argv[3]) == "1";
	static const std::string ref_file = (argc >= 5) ? argv[4] : input_file;

	std::string output_file = import_export(input_file, format, test_selection);

#ifndef SQUEY_BENCH
	std::string file_extension = input_file.substr(input_file.rfind('.') + 1);
	const auto& [args, argv_] = PVCore::PVStreamingDecompressor::executable(file_extension, PVCore::PVStreamingDecompressor::EExecType::DECOMPRESSOR);
	std::string uncompressed_file = output_file;
	std::string cmd = boost::algorithm::join(args, " ");
	if (not cmd.empty()) {
		std::string output_file2 = import_export(output_file, format, test_selection);
		output_file2 = std::filesystem::path(output_file2).make_preferred().string();
		uncompressed_file = output_file2.substr(0, output_file2.find_last_of("."));
		if (cmd == "funzip") {
#ifdef _WIN32
			cmd = "7z x -y -o" + boost::filesystem::path(output_file2).parent_path().string();
#else
			cmd = "unzip -o -qq -d " + boost::filesystem::path(output_file2).parent_path().string();
#endif
		}
		else if (cmd == "pigz -d -c") {
			cmd = "pigz -d";
		}
		else if (cmd == "zstd -d -c") {
			cmd = "zstd -d";
		}
		cmd += " " + output_file2;
		system(cmd.c_str());
		std::remove(output_file2.c_str());
	}

	bool same_content = PVRush::PVUtils::files_have_same_content(ref_file, uncompressed_file);
	std::cout << std::endl << ref_file << " - " << uncompressed_file << std::endl;
	PV_ASSERT_VALID(same_content);
	std::remove(uncompressed_file.c_str());
#endif // SQUEY_BENCH

	std::remove(output_file.c_str());
}

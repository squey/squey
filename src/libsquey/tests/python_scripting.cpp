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

UNICODE_MAIN()
{
    if (argc <= 3) {
		std::cerr << "Usage: file format ref_file" << std::endl;
		return 1;
	}

#ifdef _WIN32
	std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
	static const std::string input_file = conv.to_bytes(argv[1]);
	static const std::string format = conv.to_bytes(argv[2]);
	static const std::string ref_file =conv.to_bytes(argv[3]);
#else
	static const std::string input_file = argv[1];
	static const std::string format = argv[2];
	static const std::string ref_file = argv[3];
#endif

	std::string output_file = import_export(input_file, format, true);

    bool same_content = PVRush::PVUtils::files_have_same_content(ref_file, output_file);
	std::cout << std::endl << ref_file << " - " << output_file << std::endl;
	PV_ASSERT_VALID(same_content);
	std::remove(output_file.c_str());

	return 0;
}
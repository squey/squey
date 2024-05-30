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

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <iostream>
#include <filesystem>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cerrno>

#include "common.h"

#define FILES_DIR "../../tests/files/pvkernel/run/tickets/28/"

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " test-files-directory" << std::endl;
		return 1;
	}

	pvtest::init_ctxt();

	const QString format_path =
	    QString::fromLocal8Bit(argv[1]) + QLatin1String("/tickets/28/field_enum.format");
	const std::string& out_path = pvtest::get_tmp_filename();
	std::filesystem::copy(format_path.toStdString(), out_path);
	PVRush::PVFormat format("org", QString::fromStdString(out_path));

	int fd = open(out_path.c_str(), O_RDWR);
	std::remove(out_path.c_str());
	if (fd == -1) {
		std::cerr << "Unable to open the format for reading/writing after PVFormat::populate() : "
		          << strerror(errno) << std::endl;
		return 1;
	}
	return 0;
}

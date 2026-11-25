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

#include "../include/libpvpcap.h"

#include <libpvpcap/shell.h>
#include <rapidjson/encodings.h>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <filesystem>
#include <string_view>

#include "../include/libpvpcap/ws.h"
#include "../include/libpvpcap/exception.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include <boost/dll/runtime_symbol_info.hpp>

namespace pvpcap
{

std::string tshark_path()
{
	// Same override mechanism as SQUEY_PCAP_PROFILES_PATH in get_system_profile_dir().
	// The default locations below only work for the application itself: on Windows
	// tshark is unpacked by PVUtilitiesDecompressor from an archive sitting next to
	// squey.exe, into a cache directory named after the running executable. A test
	// binary has neither, so it needs to be told where tshark is.
	const char* tshark_env = std::getenv("SQUEY_TSHARK_PATH");
	if (tshark_env != nullptr and *tshark_env != '\0') {
		return tshark_env;
	}

#ifdef _WIN32
    return QDir(QStandardPaths::writableLocation(QStandardPaths::CacheLocation)).filePath("tshark/tshark.exe").toStdString();
#elifdef __APPLE__
    return boost::dll::program_location().parent_path().string() + "/tshark";
#else
	return "tshark";
#endif
}

std::string tshark_errors(const std::string& path)
{
	constexpr size_t most = 2000;
	std::ifstream file{std::filesystem::path(std::u8string(path.begin(), path.end()))};
	std::string said(most, '\0');
	file.read(said.data(), most);
	said.resize(size_t(file.gcount()));
	while (not said.empty() and std::isspace(static_cast<unsigned char>(said.back()))) {
		said.pop_back();
	}
	// tshark says it to whoever runs it as root, which says nothing of what went wrong. The user
	// and the group it names read "UNKNOWN" where no account has their identifiers, as in a
	// sandbox.
	const std::string_view first_line = std::string_view(said).substr(0, said.find('\n'));
	if (first_line.starts_with("Running as user \"") and
	    first_line.ends_with("This could be dangerous.")) {
		said.erase(0, first_line.size());
		while (not said.empty() and std::isspace(static_cast<unsigned char>(said.front()))) {
			said.erase(0, 1);
		}
	}
	return said;
}

} // namespace pvpcap

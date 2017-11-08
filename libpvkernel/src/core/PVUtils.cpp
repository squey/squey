/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVUtils.h>

#include <cstddef>
#include <fstream>
#include <memory>

std::string& PVCore::replace(std::string& str,
                             const std::string& from,
                             const std::string& to,
                             size_t pos /*= 0*/)
{
	// avoid an infinite loop if "from" is an empty string
	if (from.empty()) {
		return str;
	}

	while ((pos = str.find(from, pos)) != std::string::npos) {
		str.replace(pos, from.size(), to);
		// Advance to avoid replacing the same substring again
		pos += to.size();
	}

	return str;
}

std::string PVCore::file_content(const std::string& file_path)
{
	std::ifstream stream(file_path);

	return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::string PVCore::exec_cmd(const char* cmd)
{
	static constexpr const size_t MAX_LINE_SIZE = 128;

	std::array<char, MAX_LINE_SIZE> buffer;
	std::string result;
	std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
	if (not pipe)
		throw std::runtime_error(std::string("running command '") + std::string(cmd) +
		                         "'  failed!");
	while (not feof(pipe.get())) {
		if (fgets(buffer.data(), MAX_LINE_SIZE, pipe.get()) != nullptr) {
			result += buffer.data();
		}
	}
	return result;
}

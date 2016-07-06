/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVUtils.h>

#include <fstream>
#include <algorithm>

std::string& PVCore::replace(std::string& str, const std::string& from, const std::string& to)
{
	// avoid an infinite loop if "from" is an empty string
	if (from.empty()) {
		return str;
	}

	size_t pos = 0;
	while ((pos = str.find(from, pos)) != std::string::npos) {
		str.replace(pos, from.size(), to);
		// Advance to avoid replacing the same substring again
		pos += to.size();
	}

	return str;
}

size_t PVCore::row_count(const std::string& file_path)
{
	std::ifstream f(file_path);

	return std::count(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>(), '\n');
}

std::string PVCore::file_content(const std::string& file_path)
{
	std::ifstream stream(file_path);

	return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

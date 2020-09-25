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

#include <boost/algorithm/string.hpp>

#include <pvlogger.h>

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

void PVCore::remove_common_folders(std::vector<std::string>& paths)
 {
	std::vector<std::vector<std::string>> paths_folders;
	for (std::string& path : paths) {
		paths_folders.emplace_back(std::vector<std::string>());
		boost::split(paths_folders.back(), path, boost::is_any_of("/"));
	}

	size_t common_folders_depth = 0;
	bool stop = false;
	do {
		std::string current_folder;
		for (const std::vector<std::string>& path_folders : paths_folders) {
			if (current_folder.empty()) {
				current_folder = path_folders[common_folders_depth];
			}
			stop |= common_folders_depth >= path_folders.size()-1 or path_folders[common_folders_depth] != current_folder;
		}
		common_folders_depth += not stop;
	} while (not stop);
	common_folders_depth--;

	if (common_folders_depth > 0) {
		for (std::vector<std::string>& path_folders : paths_folders) {
			path_folders.erase(path_folders.begin(), path_folders.size() > common_folders_depth+1 ? path_folders.begin() + common_folders_depth+1 : path_folders.end());
		}
		for (size_t i = 0; i < paths.size(); i++) {
			paths[i] = boost::join(paths_folders[i], "/");
		}
	}
 }

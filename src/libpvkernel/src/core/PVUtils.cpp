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

#include <pvkernel/core/PVUtils.h>
#include <stdio.h>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/detail/classification.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <fstream>
#include <memory>
#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <stdexcept>

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
			paths[i] = boost::join(paths_folders[i], std::string("/"));
		}
	}
 }

 size_t PVCore::available_memory()
{
    std::string token;
    std::ifstream file("/proc/meminfo");
    while(file >> token) {
        if(token == "MemAvailable:") {
            unsigned long mem;
            if(file >> mem) {
                return mem * 1024;
            } else {
                return 0;
            }
        }
        // Ignore the rest of the line
        file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    return 0; // Nothing found
}

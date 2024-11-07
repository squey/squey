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

#include <pvkernel/filter/PVPluginsLoad.h> // for NORMALIZE_FILTER_PREFIX

#include <pvkernel/core/PVClassLibrary.h> // for PVClassLibraryLibLoader
#include <pvkernel/core/PVLogger.h>       // for PVLOG_INFO, PVLOG_WARN

#include <QString>

#include <cstdlib> // for getenv
#include <string>  // for allocator, operator+, etc

#include <boost/dll/runtime_symbol_info.hpp>

int PVFilter::PVPluginsLoad::load_all_plugins()
{
	return load_normalize_plugins();
}

int PVFilter::PVPluginsLoad::load_normalize_plugins()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(
	    QString::fromStdString(get_normalize_dir()), NORMALIZE_FILTER_PREFIX);
	if (ret == 0) {
		PVLOG_WARN("No normalization plugin have been loaded !\n");
	} else {
		PVLOG_INFO("%d normalization plugins have been loaded.\n", ret);
	}
	return ret;
}

std::string PVFilter::PVPluginsLoad::get_normalize_dir()
{
	std::string plugins_dir;
	const char* path = std::getenv("PVKERNEL_PLUGIN_PATH");
	if (path) {
		plugins_dir = std::string(path);
	}
	else {
#ifdef __APPLE__
		boost::filesystem::path exe_path = boost::dll::program_location();
		plugins_dir = exe_path.parent_path().string() + "/../PlugIns";
#else
		plugins_dir = std::string(PVKERNEL_PLUGIN_PATH);
#endif
	}

	return  plugins_dir + "/normalize-filters";
}

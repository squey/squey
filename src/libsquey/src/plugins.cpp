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

#include <string>

#include <squey/plugins.h>
#include <cstdlib>

#include <boost/dll/runtime_symbol_info.hpp>

static std::string get_squey_plugins_path()
{
	const char* path = std::getenv("SQUEY_PLUGIN_PATH");
	if (path) {
		return path;
	}
#ifdef __APPLE__
	boost::filesystem::path exe_path = boost::dll::program_location();
	return exe_path.parent_path().string() + "/../PlugIns";
#else
	return SQUEY_PLUGIN_PATH;
#endif
}

std::string squey_plugins_get_layer_filters_dir()
{
	return get_squey_plugins_path() + "/layer-filters";
}

std::string squey_plugins_get_mapping_filters_dir()
{
	return get_squey_plugins_path() + "/mapping-filters";
}

std::string squey_plugins_get_scaling_filters_dir()
{
	return get_squey_plugins_path() + "/scaling-filters";
}

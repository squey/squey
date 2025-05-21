/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef LIBSQUEY_TESTS_COMMON_H
#define LIBSQUEY_TESTS_COMMON_H

#include "test-env.h"

#include <squey/common.h>
#include <squey/PVRoot.h>
#include <squey/PVScene.h>
#include <squey/PVSource.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVPluginsLoad.h>

#include <QCoreApplication>

#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>

#define UNICODE_MAIN() \
int wmain(int argc, WCHAR **argv); \
int main(int argc, char **argv) { \
	(void) argc; \
	(void) argv; \
    LPWSTR commandLine = GetCommandLineW(); \
    int argcw = 0; \
    LPWSTR *argvw = CommandLineToArgvW(commandLine, &argcw); \
    if (!argvw) return 127; \
    int result = wmain(argcw, argvw); \
    LocalFree(argvw); \
    return result; \
} \
int wmain(int argc, WCHAR **argv)
#else
#define UNICODE_MAIN() \
int main(int argc, char **argv)
#endif

namespace pvtest
{

/**
 * Get a tmp filename not already use.
 *
 * @warning, It can be use between this call and your creation.
 */

std::string get_tmp_filename();


enum class ProcessUntil { Source, Mapped, Scaled, View };

/**
 * Create and save context for a view creation.
 *
 * * Required when we want to work with NRaw content
 */
class TestEnv
{

  public:
	/**
	 * Initialize Squey internal until a view is correctly build and return this view.
	 *
	 * dup is the number of time we want to duplicate data.
	 */
	TestEnv(std::vector<std::string> const& log_files,
	        std::string const& format_file,
	        size_t dup = 1,
	        ProcessUntil until = ProcessUntil::Source,
	        const std::string& nraw_loading_from_disk_dir = "");

	TestEnv(std::string const& log_file,
	        std::string const& format_file,
	        size_t dup = 1,
	        ProcessUntil until = ProcessUntil::Source,
	        const std::string& nraw_loading_from_disk_dir = "")
	    : TestEnv(std::vector<std::string>{log_file},
	              format_file,
	              dup,
	              until,
	              nraw_loading_from_disk_dir)
	{
	}

	Squey::PVSource& add_source(std::vector<std::string> const& log_files,
	                             std::string const& format_file,
	                             size_t dup = 1,
	                             bool new_scene = true)
	{
		return import(log_files, format_file, dup, new_scene);
	}

	Squey::PVSource& add_source(std::string const& log_file,
	                             std::string const& format_file,
	                             size_t dup = 1,
	                             bool new_scene = true)
	{
		return import(std::vector<std::string>{log_file}, format_file, dup, new_scene);
	}

	/**
	 * Clean input duplicate file at the end.
	 */
	~TestEnv()
	{
		for (auto const& path : _big_file_paths) {
			std::remove(path.c_str());
		}
	}

	/**
	 * Compute mapping assuming PVSource is valid.
	 */
	Squey::PVMapped& compute_mapping(size_t scene_id = 0, size_t src_id = 0)
	{
		const auto& scenes = root.get_children();
		assert(scene_id < scenes.size());
		auto scene_it = scenes.begin();
		std::advance(scene_it, scene_id);

		const auto& sources = (*scene_it)->get_children();
		assert(src_id < sources.size());
		auto src_it = sources.begin();
		std::advance(src_it, src_id);

		return (*src_it)->emplace_add_child();
	}

	void compute_mappings()
	{
		for (const auto& source : root.get_children<Squey::PVSource>()) {
			source->emplace_add_child();
		}
	}

	/**
	 * Compute scaling assuming PVMapped is valid.
	 */
	Squey::PVScaled&
	compute_scaling(size_t scene_id = 0, size_t src_id = 0, size_t mapped_id = 0)
	{
		// And plot the mapped values
		const auto& scenes = root.get_children();
		assert(scene_id < scenes.size());
		auto scene_it = scenes.begin();
		std::advance(scene_it, scene_id);

		const auto& sources = (*scene_it)->get_children();
		assert(src_id < sources.size());
		auto src_it = sources.begin();
		std::advance(src_it, src_id);

		const auto& mappeds = (*src_it)->get_children();
		assert(mapped_id < mappeds.size());
		auto mapped_it = mappeds.begin();
		std::advance(mapped_it, mapped_id);

		return (*mapped_it)->emplace_add_child();
	}

	void compute_scalings()
	{
		// And plot the mapped values
		for (auto* mapped : root.get_children<Squey::PVMapped>()) {
			mapped->emplace_add_child();
		}
	}

	void compute_views()
	{
		for (auto* scaled : root.get_children<Squey::PVScaled>()) {
			scaled->emplace_add_child();
		}
	}

  private:
	Squey::PVSource& import(std::vector<std::string> const& log_files,
	                         std::string const& format_file,
	                         size_t dup,
	                         bool new_scene = true,
	                         const std::string& nraw_loading_from_disk_dir = "");

  public:
	Squey::PVRoot root;

  private:
	std::vector<std::string> _big_file_paths;
};
} // namespace pvtest

#endif

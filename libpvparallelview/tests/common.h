/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_TESTS_COMMON_H
#define PVPARALLELVIEW_TESTS_COMMON_H

#include <QString>

#include "test-env.h"

#include <inendi/common.h>
#include <inendi/PVRoot.h>
#include <inendi/PVScene.h>
#include <inendi/PVSource.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvparallelview/PVParallelView.h>

#include <QCoreApplication>

namespace Inendi
{
class PVView;
}

namespace PVParallelView
{
class PVLibView;
}

inline namespace pvtest
{

bool create_plotted_table_from_args(
    Inendi::PVPlotted::plotteds_t& norm_plotted, PVRow& nrows, PVCol& ncols, int argc, char** argv);
int extra_param_start_at();
bool input_is_a_file();
void set_extra_param(int num, const char* usage_text);
void usage(const char* path);

/**
 * Get a tmp filename not already use.
 *
 * @warning, It can be use between this call and your creation.
 */
static std::string get_tmp_filename()
{
	std::string out_path;
	// Duplicate input log to make it bigger
	out_path.resize(L_tmpnam);
	// We assume that this name will not be use by another program before we
	// create it.
	tmpnam(&out_path.front());

	return out_path;
}

enum class ProcessUntil { Source, Mapped, Plotted, View };

/**
* Create and save context for a view creation.
*
* * Required when we want to work with NRaw content
*/
class TestEnv
{
  public:
	/**
	 * Initialize Inspector internal until a view is correctly build and return
	 *this view.
	 *
	 * dup is the number of time we want to duplicate data.
	 */
	TestEnv(std::vector<std::string> const& log_files,
	        std::string const& format_file,
	        size_t dup,
	        ProcessUntil until = ProcessUntil::Source,
	        const std::string& nraw_loading_from_disk_dir = "")
	{
		// Need this core application to find plugins path.
		std::string prog_name = "test_inendi";
		char* arg = const_cast<char*>(prog_name.c_str());
		int argc = 1;
		QCoreApplication app(argc, &arg);

		init_env();

		import(log_files, format_file, dup, true, nraw_loading_from_disk_dir);

		switch (until) {
		case ProcessUntil::Source:
			return;
		case ProcessUntil::Mapped:
			compute_mappings();
			return;
		case ProcessUntil::Plotted:
			compute_mappings();
			compute_plottings();
			return;
		case ProcessUntil::View:
			compute_mappings();
			compute_plottings();
			compute_views();
			return;
		}
	}

	TestEnv(std::string const& log_file,
	        std::string const& format_file,
	        size_t dup = 1,
	        ProcessUntil until = ProcessUntil::View,
	        const std::string& nraw_loading_from_disk_dir = "")
	    : TestEnv(std::vector<std::string>{log_file},
	              format_file,
	              dup,
	              until,
	              nraw_loading_from_disk_dir)
	{
	}

	Inendi::PVSource& add_source(std::vector<std::string> const& log_files,
	                             std::string const& format_file,
	                             size_t dup = 1,
	                             bool new_scene = true)
	{
		return import(log_files, format_file, dup, new_scene);
	}

	Inendi::PVSource& add_source(std::string const& log_file,
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
	Inendi::PVMapped& compute_mapping(size_t scene_id = 0, size_t src_id = 0)
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
		for (const auto& source : root.get_children<Inendi::PVSource>()) {
			source->emplace_add_child();
		}
	}

	/**
	 * Compute plotting assuming PVMapped is valid.
	 */
	Inendi::PVPlotted&
	compute_plotting(size_t scene_id = 0, size_t src_id = 0, size_t mapped_id = 0)
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

	void compute_plottings()
	{
		// And plot the mapped values
		for (auto* mapped : root.get_children<Inendi::PVMapped>()) {
			mapped->emplace_add_child();
		}
	}

	void compute_views()
	{
		for (auto* plotted : root.get_children<Inendi::PVPlotted>()) {
			plotted->emplace_add_child();
		}
	}

  private:
	Inendi::PVSource& import(std::vector<std::string> const& log_files,
	                         std::string const& format_file,
	                         size_t dup,
	                         bool new_scene = true,
	                         const std::string& nraw_loading_from_disk_dir = "")
	{

		if (dup != 1 and log_files.size() > 1) {
			throw std::runtime_error("We don't handle multiple input with duplication");
		}

		std::string new_path = log_files[0];
		if (dup > 1) {
			new_path = get_tmp_filename();
			_big_file_paths.push_back(new_path);
			std::ifstream ifs(log_files[0]);
			std::string content{std::istreambuf_iterator<char>(ifs),
			                    std::istreambuf_iterator<char>()};

			std::ofstream big_file(new_path);
			// Duplicate file to have one millions lines
			for (size_t i = 0; i < dup; i++) {
				big_file << content;
			}
		}

		PVRush::PVInputType::list_inputs inputs;

		// Input file
		QString path_file = QString::fromStdString(new_path);
		PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_file));
		inputs << file;

		for (size_t i = 1; i < log_files.size(); i++) {
			inputs << PVRush::PVInputDescription_p(
			    new PVRush::PVFileDescription(QString::fromStdString(log_files[i])));
		}

		// Load the given format file
		QString path_format = QString::fromStdString(format_file);
		PVRush::PVFormat format("format", path_format);

		// Get the source creator
		PVRush::PVSourceCreator_p sc_file;
		if (!PVRush::PVTests::get_file_sc(file, format, sc_file)) {
			throw std::runtime_error("Can't get sources.");
		}

		// Create the PVSource object
		Inendi::PVScene* scene =
		    (new_scene) ? &root.emplace_add_child("scene") : root.get_children().front();
		Inendi::PVSource& src = scene->emplace_add_child(inputs, sc_file, format);

		if (not nraw_loading_from_disk_dir.empty()) {
			src.get_rushnraw().load_from_disk(nraw_loading_from_disk_dir);
		} else {
			PVRush::PVControllerJob_p job = src.extract(0);
			src.wait_extract_end(job);
		}
		return src;
	}

  public:
	Inendi::PVRoot root;

	PVParallelView::PVLibView* get_lib_view()
	{
		return PVParallelView::common::get_lib_view(*root.current_view());
	}

  private:
	std::vector<std::string> _big_file_paths;
};
}

#endif

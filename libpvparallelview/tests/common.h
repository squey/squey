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
#include <inendi/PVView.h>
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

namespace PVParallelView
{
class PVLibView;
}

bool create_plotted_table_from_args(Inendi::PVPlotted::uint_plotted_table_t& norm_plotted,
                                    PVRow& nrows,
                                    PVCol& ncols,
                                    int argc,
                                    char** argv);
int extra_param_start_at();
bool input_is_a_file();
void set_extra_param(int num, const char* usage_text);
void usage(const char* path);
Inendi::PVView_sp& get_view_sp();

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
	TestEnv(std::string const& log_file, std::string const& format_file, size_t dup = 1)
	    : _big_file_path(get_tmp_filename())
	{
		// Need this core application to find plugins path.
		std::string prog_name = "test_inendi";
		char* arg = const_cast<char*>(prog_name.c_str());
		int argc = 1;
		QCoreApplication app(argc, &arg);

		init_env();

		// Load plugins to fill the nraw
		PVFilter::PVPluginsLoad::load_all_plugins(); // Splitters
		PVRush::PVPluginsLoad::load_all_plugins();   // Sources
		Inendi::common::load_mapping_filters();
		Inendi::common::load_plotting_filters();

		// Initialize sse4 detection
		PVCore::PVIntrinsics::init_cpuid();

		{
			std::ifstream ifs(log_file);
			std::string content{std::istreambuf_iterator<char>(ifs),
			                    std::istreambuf_iterator<char>()};

			std::ofstream big_file(_big_file_path);
			// Duplicate file to have one millions lines
			for (size_t i = 0; i < dup; i++) {
				big_file << content;
			}
		}

		// Input file
		QString path_file = QString::fromStdString(_big_file_path);
		PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_file));

		// Load the given format file
		QString path_format = QString::fromStdString(format_file);
		PVRush::PVFormat format("format", path_format);
		if (!format.populate()) {
			throw std::runtime_error("Can't read format file " + format_file);
		}

		// Get the source creator
		PVRush::PVSourceCreator_p sc_file;
		if (!PVRush::PVTests::get_file_sc(file, format, sc_file)) {
			throw std::runtime_error("Can't get sources.");
		}

		// Create the PVSource object
		Inendi::PVScene& scene = root.emplace_add_child("scene");
		Inendi::PVSource& src =
		    scene.emplace_add_child(PVRush::PVInputType::list_inputs() << file, sc_file, format);
		PVRush::PVControllerJob_p job = src.extract();
		job->wait_end();

		Inendi::PVMapped& mapped = src.emplace_add_child();

		Inendi::PVPlotted& plotted = mapped.emplace_add_child();

		view = &plotted.emplace_add_child();
	}

	PVParallelView::PVLibView* get_lib_view()
	{
		return PVParallelView::common::get_lib_view(*view);
	}

  private:
	Inendi::PVView* view;
	Inendi::PVRoot root;
	std::string _big_file_path;
};

#endif

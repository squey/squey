#ifndef LIBINENDI_TESTS_COMMON_H
#define LIBINENDI_TESTS_COMMON_H

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

#include <QCoreApplication>

namespace pvtest
{

/**
 * Get a tmp filename not already use.
 *
 * @warning, It can be use between this call and your creation.
 */
std::string get_tmp_filename()
{
	std::string out_path;
	// Duplicate input log to make it bigger
	out_path.resize(L_tmpnam);
	// We assume that this name will not be use by another program before we create it.
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
	 * Initialize Inspector internal until a view is correctly build and return this view.
	 *
	 * dup is the number of time we want to duplicate data.
	 */
	TestEnv(std::string const& log_file, std::string const& format_file, size_t dup = 1)
	    : root(new Inendi::PVRoot()), _big_file_path(get_tmp_filename())
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

		import(log_file, format_file, dup);
	}

	void add_source(std::string const& log_file, std::string const& format_file, size_t dup = 1)
	{
		import(log_file, format_file, dup);
	}

	/**
	 * Clean input duplicate file at the end.
	 */
	~TestEnv() { std::remove(_big_file_path.c_str()); }

	/**
	 * Compute mapping assuming PVSource is valid.
	 */
	Inendi::PVMapped_p compute_mapping(int index = 0)
	{
		const auto& sources = root->get_children<Inendi::PVSource>();
		assert(index < sources.size());
		Inendi::PVMapped_p mapped = sources[index]->emplace_add_child();
		mapped->process_from_parent_source();
		return mapped;
	}

	void compute_mappings()
	{
		for (const auto& source : root->get_children<Inendi::PVSource>()) {
			Inendi::PVMapped_p mapped = source->emplace_add_child();
			mapped->process_from_parent_source();
		}
	}

	/**
	 * Compute plotting assuming PVMapped is valid.
	 */
	Inendi::PVPlotted_p compute_plotting(int index = 0)
	{
		// And plot the mapped values
		const auto& mappeds = root->get_children<Inendi::PVMapped>();
		assert(index < mappeds.size());
		Inendi::PVPlotted_p plotted = mappeds[index]->emplace_add_child();
		plotted->process_from_parent_mapped();
		return plotted;
	}

	void compute_plottings()
	{
		// And plot the mapped values
		for (const auto& mapped : root->get_children<Inendi::PVMapped>()) {
			Inendi::PVPlotted_p plotted = mapped->emplace_add_child();
			plotted->process_from_parent_mapped();
		}
	}

  private:
	void import(std::string const& log_file, std::string const& format_file, size_t dup)
	{

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
		Inendi::PVScene_p scene = root->emplace_add_child("scene");
		Inendi::PVSource_sp src =
		    scene->emplace_add_child(PVRush::PVInputType::list_inputs() << file, sc_file, format);
		PVRush::PVControllerJob_p job = src->extract();
		job->wait_end();
	}

  public:
	Inendi::PVRoot_p root;

  private:
	std::string _big_file_path;
};
}

#endif

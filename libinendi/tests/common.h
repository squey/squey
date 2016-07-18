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
	 * Initialize Inspector internal until a view is correctly build and return this view.
	 *
	 * dup is the number of time we want to duplicate data.
	 */
	TestEnv(std::string const& log_file,
	        std::string const& format_file,
	        size_t dup,
	        ProcessUntil until = ProcessUntil::Source)
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

		import(log_file, format_file, dup);

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

	Inendi::PVSource& add_source(std::string const& log_file,
	                             std::string const& format_file,
	                             size_t dup = 1,
	                             bool new_scene = true)
	{
		return import(log_file, format_file, dup, new_scene);
	}

	/**
	 * Clean input duplicate file at the end.
	 */
	~TestEnv() { std::remove(_big_file_path.c_str()); }

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
	Inendi::PVSource& import(std::string const& log_file,
	                         std::string const& format_file,
	                         size_t dup,
	                         bool new_scene = true)
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
		Inendi::PVScene* scene =
		    (new_scene) ? &root.emplace_add_child("scene") : root.get_children().front();
		Inendi::PVSource& src =
		    scene->emplace_add_child(PVRush::PVInputType::list_inputs() << file, sc_file, format);
		PVRush::PVControllerJob_p job = src.extract(0);
		job->wait_end();
		return src;
	}

  public:
	Inendi::PVRoot root;

  private:
	std::string _big_file_path;
};
}

#endif

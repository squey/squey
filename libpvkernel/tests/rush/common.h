#ifndef LIBPVKERNEL_RUSH_TESTS_COMMON_H
#define LIBPVKERNEL_RUSH_TESTS_COMMON_H

#include "test-env.h"
#include "helpers.h"

#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVUnicodeSource.h>

#include <QCoreApplication>

#include <functional>
#include <omp.h>

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
 * Prepare Context to run tests.
 *
 * * Set environment variables
 * * Prepare QCoreApplication
 * * Load plugins
 * * Init cpu features
 *
 * @note : we use constructor attribute to make sure every test which include this
 * file have correctly initialized environment.
 */
__attribute__((constructor)) void init_ctxt()
{
	// Need this core application to find plugins path.
	std::string prog_name = "test_pvkernel_rush";
	char* arg = const_cast<char*>(prog_name.c_str());
	int argc = 1;
	QCoreApplication app(argc, &arg);

	init_env();

	// Load plugins to fill the nraw
	PVFilter::PVPluginsLoad::load_all_plugins(); // Splitters
	PVRush::PVPluginsLoad::load_all_plugins();   // Sources

	// Initialize sse4 detection
	PVCore::PVIntrinsics::init_cpuid();
}

/**
 * Duplicate input log dup times and return the new file with these data.
 */
std::string duplicate_log_file(std::string const& log_file, size_t dup)
{
	std::ifstream ifs(log_file);
	std::string content{std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>()};

	std::string big_log_file = get_tmp_filename();
	std::ofstream big_file(big_log_file);
	// Duplicate file to have one millions lines
	for (size_t i = 0; i < dup; i++) {
		big_file << content;
	}
	return big_log_file;
}

/**
 * Create sources and splitter to be read for splitting.
 *
 * Usefull to check splitter behavior.
 */
class TestSplitter
{
  public:
	TestSplitter(std::string const& log_file, size_t dup = 1)
	    : _big_file_path(duplicate_log_file(log_file, dup))
	    , _source(std::make_shared<PVRush::PVInputFile>(_big_file_path.c_str()), chunk_size)
	{
	}

	~TestSplitter() { std::remove(_big_file_path.c_str()); }

	std::tuple<size_t, size_t, std::string>
	run_normalization(PVFilter::PVChunkFilterByElt const& flt_f)
	{
		std::string output_file = get_tmp_filename();
		// Extract source and split fields.
		std::ofstream ofs(output_file);

		size_t nelts_org = 0;
		size_t nelts_valid = 0;
		double duration = 0.;

		std::vector<PVCore::PVChunk*> _chunks;
		while (PVCore::PVChunk* pc = _source()) {
			_chunks.push_back(pc);
		}

// TODO : Parallelism slow down splitting. It looks like it is a locally issue on
// function splitter object with bad managed memory.
#pragma omp parallel reduction(+ : nelts_org, nelts_valid, duration)
		{
			std::ostringstream oss;
#pragma omp for nowait
			for (auto it = _chunks.begin(); it < _chunks.end(); ++it) {
				PVCore::PVChunk* pc = *it;
				auto start = std::chrono::steady_clock::now();
				flt_f(pc);
				std::chrono::duration<double> dur(std::chrono::steady_clock::now() - start);
				duration += dur.count();
				size_t no = 0;
				size_t nv = 0;
				pc->get_elts_stat(no, nv);
				nelts_org += no;
				nelts_valid += nv;
				dump_chunk_csv(*pc, oss);
				pc->free();
			}

#pragma omp for ordered
			for (int i = 0; i < omp_get_num_threads(); i++) {
#pragma omp ordered
				ofs << oss.str();
			}
		}
		std::cout << duration;
		return std::make_tuple(nelts_org, nelts_valid, output_file);
	}

  private:
	static constexpr size_t chunk_size = 6000;

	std::string _big_file_path;
	PVRush::PVUnicodeSource<> _source;
};

/**
 * Create and save context for a view creation.
 *
 * * Required when we want to work with NRaw content
 */
class TestEnv
{

  public:
	/**
	 * Initialize Inspector internal until pipeline is ready to process inputs.
	 *
	 * NRaw is not loaded, it has to be done with the load_data methods.
	 */
	TestEnv(std::string const& log_file,
	        std::string const& format_file,
	        size_t dup = 1,
	        std::string const& extra_input = "")
	    : _format("format", QString::fromStdString(format_file))
	    , _big_file_path(duplicate_log_file(log_file, dup))
	{

		if (dup != 1 and extra_input != "") {
			throw std::runtime_error("We don't handle mutliple input with duplication");
		}

		// Load the given format file
		if (!_format.populate()) {
			throw std::runtime_error("Can't read format file " + format_file);
		}

		std::vector<std::string> filenames{log_file};
		if (extra_input != "") {
			filenames.push_back(extra_input);
		}

		for (std::string const& filename : filenames) {
			// Input file
			QString path_file = QString::fromStdString(filename);
			PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_file));

			// Get the source creator
			PVRush::PVSourceCreator_p sc_file;
			if (!PVRush::PVTests::get_file_sc(file, _format, sc_file)) {
				throw std::runtime_error("Can't get sources.");
			}

			// Process that file with the found source creator thanks to the extractor
			PVRush::PVSourceCreator::source_p src =
			    sc_file->create_source_from_input(file, _format);
			if (!src) {
				throw std::runtime_error("Unable to create PVRush source from file " + log_file +
				                         "\n");
			}

			// Create the extractor
			_ext.add_source(src);
		}
		_ext.set_format(_format);
		_ext.set_chunk_filter(_format.create_tbb_filters());
	}

	void load_data(size_t begin = 0)
	{
		PVRush::PVControllerJob_p job = _ext.process_from_agg_nlines(begin);
		job->wait_end();
	}

	/**
	 * Clean up duplicated file when it is over.
	 */
	~TestEnv() { std::remove(_big_file_path.c_str()); }

	/**
	 * Get number of row in the imported NRaw.
	 */
	size_t get_nraw_size() const { return _ext.get_nraw().get_row_count(); }

	PVRush::PVExtractor _ext;

  private:
	PVRush::PVFormat _format;
	std::string _big_file_path;
};
}

#endif

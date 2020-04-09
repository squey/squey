#ifndef LIBPVKERNEL_RUSH_TESTS_COMMON_H
#define LIBPVKERNEL_RUSH_TESTS_COMMON_H

#include "test-env.h"
#include "helpers.h"

#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVNraw.h>
#include <pvkernel/rush/PVUnicodeSource.h>

#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVPluginsLoad.h>

#include <QCoreApplication>

#include <functional>
#include <omp.h>
#include <sstream>

namespace pvtest
{

/**
 * Get a tmp filename not already use.
 *
 * @warning, It can be use between this call and your creation.
 */
std::string get_tmp_filename()
{
	char buffer[L_tmpnam];
	return tmpnam(buffer);
}

/**
 * Prepare Context to run tests.
 *
 * * Set environment variables
 * * Prepare QCoreApplication
 * * Load plugins
 * * Init cpu features
 */
void init_ctxt()
{
	// Need this core application to find plugins path.
	std::string prog_name = "test_pvkernel_rush";
	char* arg = const_cast<char*>(prog_name.c_str());
	int argc = 1;
	QCoreApplication app(argc, &arg);

	init_env();
}

/**
 * Duplicate input log dup times and return the new file with these data.
 */
std::string duplicate_log_file(std::string const& log_file, size_t dup)
{
	if (dup == 1) {
		return log_file;
	}

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
	TestSplitter(std::string const& log_file = "", size_t dup = 1)
	{
		init_ctxt();

		if (log_file.size() != 0) {
			reset(log_file, dup);
		}
	}

	~TestSplitter()
	{
		if (_need_cleanup)
			std::remove(_big_file_path.c_str());
	}

	std::tuple<size_t, size_t, std::string>
	run_normalization(PVFilter::PVChunkFilterByElt const& flt_f)
	{
		if (_source.get() == nullptr) {
			throw std::runtime_error("source not created");
		}

		std::string output_file = get_tmp_filename();
		// Extract source and split fields.
		std::ofstream ofs(output_file);

		size_t nelts_org = 0;
		size_t nelts_valid = 0;
		double duration = 0.;

		std::vector<PVCore::PVTextChunk*> _chunks;
		while (PVCore::PVTextChunk* pc = (*_source.get())()) {
			_chunks.push_back(pc);
		}

#pragma omp parallel reduction(+ : nelts_org, nelts_valid) reduction(max : duration)
		{
			std::ostringstream oss;
			double local_duration = 0.;
#pragma omp for nowait
			for (auto it = _chunks.begin(); it < _chunks.end(); ++it) {
				PVCore::PVTextChunk* pc = *it;
				auto start = std::chrono::steady_clock::now();
				flt_f(pc);
				std::chrono::duration<double> dur(std::chrono::steady_clock::now() - start);
				local_duration += dur.count();
				size_t no = 0;
				size_t nv = 0;
				pc->get_elts_stat(no, nv);
				nelts_org += no;
				nelts_valid += nv;
				dump_chunk_csv(*pc, oss);
				pc->free();
			}
			duration = local_duration;

#pragma omp for ordered
			for (int i = 0; i < omp_get_num_threads(); i++) {
#pragma omp ordered
				ofs << oss.str();
			}
		}
		std::cout << duration;
		return std::make_tuple(nelts_org, nelts_valid, output_file);
	}

	void reset(std::string const& log_file, size_t dup = 1)
	{
		_big_file_path = duplicate_log_file(log_file, dup);
		_source.reset(new PVRush::PVUnicodeSource<>(
		    std::make_shared<PVRush::PVInputFile>(_big_file_path.c_str()), chunk_size));
		_need_cleanup = dup > 1;
	}

  private:
	static constexpr size_t chunk_size = 6000;

	std::string _big_file_path;
	std::unique_ptr<PVRush::PVUnicodeSource<>> _source;
	bool _need_cleanup;
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
	TestEnv(std::string const& log_file, std::string const& format_file, size_t dup = 1)
	    : TestEnv(std::vector<std::string>(1, log_file), format_file, dup)
	{
	}

	TestEnv(std::vector<std::string> const& log_files,
	        std::string const& format_file,
	        size_t dup = 1)
	{
		init_ctxt();
		reset(log_files, format_file, dup);
	}

	void load_data(size_t begin = 0)
	{
		PVRush::PVExtractor ext(_format, *_nraw_output.get(), _sc_file, _list_inputs);
		PVRush::PVControllerJob_p job = ext.process_from_agg_nlines(begin);
		job->wait_end();
	}

	/**
	 * Clean up duplicated file when it is over.
	 */
	~TestEnv()
	{
		if (_need_cleanup)
			std::remove(_big_file_path.c_str());
	}

	void
	reset(std::vector<std::string> const& log_files, std::string const& format_file, size_t dup = 1)
	{
		_format = PVRush::PVFormat("format", QString::fromStdString(format_file));
		_nraw_output.reset(new PVRush::PVNrawOutput(_nraw));
		_big_file_path = duplicate_log_file(log_files[0], dup);
		_need_cleanup = (dup > 1);

		if (dup != 1 and log_files.size() > 1) {
			throw std::runtime_error("We don't handle mutliple input with duplication");
		}

		_list_inputs << PVRush::PVInputDescription_p(
		    new PVRush::PVFileDescription(QString::fromStdString(_big_file_path)));
		for (size_t i = 1; i < log_files.size(); i++) {
			// Input file
			QString path_file = QString::fromStdString(log_files[i]);
			_list_inputs << PVRush::PVInputDescription_p(new PVRush::PVFileDescription(path_file));
		}

		// Get the source creator
		if (!PVRush::PVTests::get_file_sc(_list_inputs.front(), _format, _sc_file)) {
			throw std::runtime_error("Can't get sources.");
		}
	}

	/**
	 * Get number of row in the imported NRaw.
	 */
	size_t get_nraw_size() const { return _nraw.row_count(); }

	PVRush::PVFormat _format;
	PVRush::PVNraw _nraw;
	std::unique_ptr<PVRush::PVNrawOutput> _nraw_output;
	QList<std::shared_ptr<PVRush::PVInputDescription>> _list_inputs;
	PVRush::PVSourceCreator_p _sc_file;

  private:
	std::string _big_file_path;
	bool _need_cleanup;
};
} // namespace pvtest

#endif

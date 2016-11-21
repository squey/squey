/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <pvkernel/core/PVColumnIndexes.h> // for PVColumnIndexes
#include <pvkernel/core/PVExporter.h>
#include <pvkernel/core/PVSelBitField.h> // for PVSelBitField

#include <pvbase/types.h> // for PVRow

#include <cassert>    // for assert
#include <cstddef>    // for size_t
#include <functional> // for function
#include <omp.h>      // for omp_get_thread_num
#include <ostream>    // for flush, ostream
#include <string>     // for allocator, string, etc
#include <sys/stat.h> // for mkfifo
#include <cstring>    // for std::strerror
#include <atomic>

#include <fcntl.h>
#include <sys/wait.h>

#include <tbb/pipeline.h>

const std::string PVCore::PVExporter::default_sep_char = ",";
const std::string PVCore::PVExporter::default_quote_char = "\"";

enum { INPUT, OUTPUT };

const std::unordered_map<size_t, std::pair<std::string, std::string>>
    PVCore::PVExporter::_compressors = {{(size_t)CompressionType::NONE, {}},
                                        {(size_t)CompressionType::GZ, {"pigz", ".gz"}},
                                        {(size_t)CompressionType::BZ2, {"lbzip2", ".bz2"}},
                                        {(size_t)CompressionType::ZIP, {"zip", ".zip"}}};

PVCore::PVExporter::PVExporter(const std::string& file_path,
                               const PVCore::PVSelBitField& sel,
                               const PVCore::PVColumnIndexes& column_indexes,
                               PVRow step_count,
                               const export_func& f,
                               CompressionType compression_type /* = CompressionType::NONE */,
                               const std::string& sep_char /* = default_sep_char */,
                               const std::string& quote_char, /* = default_quote_char */
                               const std::string& header      /* = std::string() */
                               )
    : _file_path(file_path)
    , _sel(sel)
    , _column_indexes(column_indexes)
    , _step_count(step_count)
    , _compression_type(compression_type)
    , _sep_char(sep_char)
    , _quote_char(quote_char)
    , _f(f)
{
	assert(_column_indexes.size() != 0);

	init();

	if (not header.empty()) {
		write(_fd, header.c_str(), header.size());
	}
}

PVCore::PVExporter::~PVExporter()
{
	if (not _finished) {
		pvlogger::error()
		    << "PVCore::PVExporter::wait_finished() not called before object destruction"
		    << std::endl;
		assert(false);
	}
}

void PVCore::PVExporter::export_rows(size_t start_index)
{
	const size_t thread_count = std::thread::hardware_concurrency();

	int thread_index = -1;

	tbb::parallel_pipeline(
	    thread_count /* = max_number_of_live_token */,
	    tbb::make_filter<void, std::pair<size_t, size_t>>(
	        tbb::filter::serial_in_order,
	        [&](tbb::flow_control& fc) -> std::pair<size_t, size_t> {
		        if ((size_t)++thread_index == thread_count) {
			        fc.stop();
		        }

		        const size_t range = _step_count / thread_count;
		        const size_t begin_index = start_index + (thread_index * range);
		        const size_t len =
		            (size_t)thread_index == (thread_count - 1) ? _step_count - begin_index : range;
		        const size_t end_index = begin_index + len;

		        return std::make_pair(begin_index, end_index);
		    }) &
	        tbb::make_filter<std::pair<size_t, size_t>, std::string>(
	            tbb::filter::parallel,
	            [&](const std::pair<size_t, size_t>& range) -> std::string {

		            const size_t begin_index = range.first;
		            const size_t end_index = range.second;

		            std::string content;
		            for (PVRow row_index = begin_index; row_index < end_index; row_index++) {

			            if (!_sel.get_line_fast(row_index)) {
				            continue;
			            }

			            content += _f(row_index, _column_indexes, _sep_char, _quote_char) + "\n";
		            }

		            return content;

		        }) &
	        tbb::make_filter<std::string, void>(
	            tbb::filter::serial_in_order, [&](const std::string& content) {
		            /*
		             * Avoid potential deadlock when writing content
		             */
		            if (_compression_type != CompressionType::NONE and _compression_status == 0) {
			            int status = 0;
			            waitpid(_compression_pid, &status, WNOHANG | WUNTRACED);
			            if (WIFEXITED(status)) {
				            _compression_status = WEXITSTATUS(status);
			            }
		            }

		            if (_compression_status == 0) {
			            if (write(_fd, content.c_str(), content.size()) < (int)content.size()) {
				            std::string error_msg = std::strerror(errno);
				            close(_fd);
				            throw PVCore::PVExportError(
				                std::string(
				                    "Export failed with the following error message :\n\n") +
				                error_msg);
			            }
		            }
		        }));
}

void PVCore::PVExporter::init()
{
	if (_compression_type == CompressionType::NONE) {
		_fd = open(_file_path.c_str(), O_CREAT | O_WRONLY, 0666);
		if (_fd == -1) {
			throw PVCore::PVExportError("Unable to create file '" + _file_path + "'");
		}
		return;
	}

	// Used to forward data to compressor
	int in_pipe[2];
	pipe(in_pipe);

	// Used to get error message back from compressor
	int out_pipe[2];
	pipe(out_pipe);

	/**
	 * We need to spawn a new process to have unshared file descriptors
	 */
	_compression_pid = fork();
	switch (_compression_pid) {
	case 0: { // child process
		// redirect parent pipe to std::in
		close(in_pipe[OUTPUT]);
		dup2(in_pipe[INPUT], STDIN_FILENO);

		// redirect std::err to parent
		close(out_pipe[INPUT]);
		dup2(out_pipe[OUTPUT], STDERR_FILENO);

		// redirect std::out to file
		_compression_fd = open(_file_path.c_str(), O_CREAT | O_WRONLY, 0666);
		if (_compression_fd == -1) {
			std::cerr << "Unable to create file '" + _file_path + "'" << std::endl;
			close(in_pipe[INPUT]);
			exit(-1);
		}
		dup2(_compression_fd, STDOUT_FILENO);

		const std::string& cmd = executable(_compression_type);
		if (execlp(cmd.c_str(), cmd.c_str(), (char*)nullptr) == -1) {
			_exit(-1);
		}
	}
	default: { // parent process
		close(in_pipe[INPUT]);
		_fd = in_pipe[OUTPUT];

		close(out_pipe[OUTPUT]);
		_compression_error_fd = out_pipe[INPUT];
	}
	}
}

void PVCore::PVExporter::wait_finished()
{
	if (_finished) {
		return;
	}

	close(_fd);

	if (_compression_type == CompressionType::NONE) {
		_finished = true;
		return;
	}

	close(_compression_fd);

	int compression_status;
	pid_t pid = waitpid(_compression_pid, &compression_status, 0);

	_finished = true;

	// throw exception with error message if compression failed
	if (_compression_status != 0 or (pid > 0 && compression_status != 0)) {
		std::string error_msg;
		char buffer[1024];

		while (true) {
			int read_count = read(_compression_error_fd, buffer, sizeof(buffer));
			if (read_count <= 0) {
				break;
			}
			error_msg += std::string(buffer, 0, read_count);
		}

		close(_compression_error_fd);

		throw PVCore::PVExportError("Compression failed with the following error message :\n\n" +
		                            error_msg);
	}
}

const std::string& PVCore::PVExporter::extension(PVExporter::CompressionType compression_type)
{
	return PVCore::PVExporter::_compressors.at((size_t)compression_type).second;
}

std::string PVCore::PVExporter::executable(PVExporter::CompressionType compression_type)
{
	return "/usr/bin/" + PVCore::PVExporter::_compressors.at((size_t)compression_type).first;
}

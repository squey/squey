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

#ifndef _PVCORE_PVSTREAMINGCOMPRESSOR_H__
#define _PVCORE_PVSTREAMINGCOMPRESSOR_H__

#include <pvkernel/core/PVOrderedMap.h>
#include <sys/types.h>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <stdexcept>
#include <tuple>
#include <utility>
#ifdef _WIN32
#include <windows.h>
#endif

namespace PVCore
{
template <class Key, class Value> class PVOrderedMap;

namespace __impl
{

/**
 * This base classe is used for streaming compression and decompression
 * using pipes and external binaries.
 *
 * Selected compression/decompression tool is based on file extension.
 * In case file extension is not supported, fallback as passthrough.
 */
class PVStreamingBase
{
  protected:
	static const PVCore::PVOrderedMap<std::string, std::pair<std::string, std::string>>
	    _supported_compressors;

  public:
  enum class EExecType {
	COMPRESSOR,
	DECOMPRESSOR
  };

  public:
	PVStreamingBase(const std::string& path);
	virtual ~PVStreamingBase();

  public:
	void cancel();
	void wait_finished();

  public:
	static std::vector<std::string> supported_extensions();
	const std::string& path() const { return _path; }

  public:
	static std::tuple<std::vector<std::string>, std::vector<char*>> executable(const std::string& extension, EExecType type, const std::string& output_name = "-");

  protected:
	int return_status(std::string* status_message = nullptr);
	virtual void do_wait_finished() = 0;

  protected:
	std::string _path;
	int _fd = -1;
	int _status_code = 0;
	std::string _status_msg;
	std::string _extension;
	bool _passthrough;
	bool _canceled = false;
	bool _finished = false;

	int _status_fd = -1;
	int _output_fd = -1;
	std::vector<std::string> _args;
#if defined(__linux__) || defined(__APPLE__)
	pid_t _child_pid = -1;
	std::vector<char*> _argv;
#elifdef _WIN32
	HANDLE _child_pid = INVALID_HANDLE_VALUE;
	std::wstring _cmdline;
#else
#error "Unsupported platform"
#endif
};

} // namespace __impl

struct PVStreamingCompressorError : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

class PVStreamingCompressor : public __impl::PVStreamingBase
{
  public:
	PVStreamingCompressor(const std::string& path);
	~PVStreamingCompressor();

  public:
	void write(const std::string& content);

  private:
	void do_wait_finished() override;
};

struct PVStreamingDecompressorError : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

class PVStreamingDecompressor : public __impl::PVStreamingBase
{
  public:
	using chunk_sizes_t = std::pair<size_t /*uncompressed*/, size_t /*compressed*/>;

  public:
	PVStreamingDecompressor(const std::string& path);
	~PVStreamingDecompressor();

  public:
	chunk_sizes_t read(char* buffer, size_t count);
	void reset();

  private:
	void init();
	void wait_finished() { return __impl::PVStreamingBase::wait_finished(); }
	void do_wait_finished() override;

  private:
	std::thread _thread;
	std::atomic<size_t> _compressed_chunk_size;
	bool _init = false;
	int _write_fd = -1;
};

} // namespace PVCore

#endif // _PVCORE_PVSTREAMINGCOMPRESSOR_H__

/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef _PVCORE_PVSTREAMINGCOMPRESSOR_H__
#define _PVCORE_PVSTREAMINGCOMPRESSOR_H__

#include <pvkernel/core/PVOrderedMap.h>

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace PVCore
{

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
	static const std::string BINARIES_PATH;

  public:
	PVStreamingBase(const std::string& path);
	virtual ~PVStreamingBase();

  public:
	void cancel();
	void wait_finished();

  public:
	static std::vector<std::string> supported_extensions();
	const std::string& path() const { return _path; }

  protected:
	int return_status(std::string* status_message = nullptr);
	virtual void do_wait_finished() = 0;

  protected:
	std::string _path;
	int _fd = -1;
	pid_t _child_pid = -1;
	int _status_fd = -1;
	int _status_code = 0;
	std::string _status_msg;
	std::string _extension;
	bool _passthrough;
	bool _canceled = false;
	bool _finished = false;
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

  public:
	static std::string executable(const std::string& extension);

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

  public:
	static std::string executable(const std::string& extension);

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

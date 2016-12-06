/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVINPUTFILE_FILE_H
#define PVINPUTFILE_FILE_H

#include <pvkernel/core/PVStreamingCompressor.h>
#include <pvkernel/rush/PVInput.h> // for PVInputException, etc

#include <atomic>
#include <cstddef> // for size_t
#include <cstdint> // for uint64_t
#include <cstring> // for strerror
#include <fstream> // for ifstream
#include <string>  // for allocator, operator+, etc
#include <thread>

#include <QString>

namespace PVRush
{

class PVInputFile : public PVInput
{
  public:
	explicit PVInputFile(const char* path);
	PVInputFile(const PVInputFile& /*org*/) = delete;
	~PVInputFile() override;

  public:
	chunk_sizes_t operator()(char* buffer, size_t n) override;
	void cancel() override;
	void seek_begin() override;
	QString human_name() override;

  public:
	// File specific
	uint64_t file_size();

  private:
	std::string _path;
	PVCore::PVStreamingDecompressor _decompressor;

	CLASS_INPUT(PVRush::PVInputFile)
};

class PVInputFileOpenException : public PVInputException
{
  public:
	PVInputFileOpenException(std::string const& path, int err)
	    : PVInputException("Unable to open file " + path + ": " + strerror(err))
	{
	}
};
} // namespace PVRush

#endif

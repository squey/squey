/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVINPUTFILE_FILE_H
#define PVINPUTFILE_FILE_H

#include <pvkernel/rush/PVInput.h>
#include <fstream>
#include <string>

namespace PVRush
{

class PVInputFile : public PVInput
{
  public:
	PVInputFile(const char* path);
	PVInputFile(const PVInputFile& /*org*/) = delete;
	~PVInputFile();

  public:
	size_t operator()(char* buffer, size_t n) override;
	input_offset current_input_offset() override;
	void seek_begin() override;
	QString human_name() override;

  public:
	// File specific
	uint64_t file_size();

  protected:
	std::ifstream _file;
	std::string _path;

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
}

#endif

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
	void release() override;

  public:
	size_t operator()(char* buffer, size_t n) override;
	input_offset current_input_offset() override;
	void seek_begin() override;
	bool seek(input_offset off) override;
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
	PVInputFileOpenException(const char* path, int err) : _path(path), _err(err)
	{
		_what = "Unable to open file ";
		_what += _path;
		_what += ": ";
		_what += strerror(err);
	}

  public:
	inline int err() const { return _err; }
	inline std::string const& path() const { return _path; }
	std::string const& what() const { return _what; }

  protected:
	std::string _path;
	int _err;
	std::string _what;
};
}

#endif

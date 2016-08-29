/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVSERIALIZEARCHIVEEXCEPTIONS_H
#define PVCORE_PVSERIALIZEARCHIVEEXCEPTIONS_H

#include <string>
#include <stdexcept>

namespace PVCore
{

class PVSerializeArchiveError : public std::runtime_error
{
	using std::runtime_error::runtime_error;
};

class PVSerializeArchiveErrorNoObject : public PVSerializeArchiveError
{
	using PVSerializeArchiveError::PVSerializeArchiveError;
};

class PVSerializeArchiveErrorFileNotReadable : public PVSerializeArchiveError
{
  public:
	PVSerializeArchiveErrorFileNotReadable(std::string const& path)
	    : PVSerializeArchiveError("File " + path + " does not exist or is not readable.")
	    , _path(path)
	{
	}

  public:
	std::string const& get_path() const { return _path; }

  protected:
	std::string _path;
};
}

#endif

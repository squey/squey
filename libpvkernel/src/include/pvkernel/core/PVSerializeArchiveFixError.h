/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVSERIALIZEARCHIVEFIXERROR
#define PVCORE_PVSERIALIZEARCHIVEFIXERROR

#include <memory>
#include <stdexcept>
#include <string>

namespace PVCore
{

class PVSerializeObject;
class PVSerializeArchiveError;

class PVSerializeReparaibleError : public std::runtime_error
{
  public:
	PVSerializeReparaibleError(std::string const& what, std::string path)
	    : std::runtime_error(what), _path(std::move(path))
	{
	}

	std::string const& logical_path() const { return _path; }

  private:
	std::string _path;
};

class PVSerializeReparaibleFileError : public PVSerializeReparaibleError
{
  public:
	PVSerializeReparaibleFileError(std::string const& what,
	                               std::string const& path,
	                               std::string value)
	    : PVSerializeReparaibleError(what, path), _value(std::move(value))
	{
	}

	std::string const& old_value() const { return _value; }

  private:
	std::string _value;
};

class PVSerializeReparaibleCredentialError : public PVSerializeReparaibleError
{
  public:
	using PVSerializeReparaibleError::PVSerializeReparaibleError;
};
} // namespace PVCore

#endif

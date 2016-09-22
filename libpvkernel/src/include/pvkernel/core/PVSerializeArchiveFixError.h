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

#include <QVariant>

namespace PVCore
{

class PVSerializeObject;
class PVSerializeArchiveError;

class PVSerializeReparaibleError : public std::runtime_error
{
  public:
	PVSerializeReparaibleError(std::string const& what, std::string path, std::string value)
	    : std::runtime_error(what), _path(std::move(path)), _value(std::move(value))
	{
	}

	std::string const& old_value() const { return _value; }
	std::string const& logical_path() const { return _path; }

  private:
	std::string _path;
	std::string _value;
};
} // namespace PVCore

#endif

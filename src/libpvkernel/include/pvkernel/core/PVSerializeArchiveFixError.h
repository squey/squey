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

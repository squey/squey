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

#ifndef __PVKERNEL_CORE_PVVERSION_H__
#define __PVKERNEL_CORE_PVVERSION_H__

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <assert.h>
#include <stddef.h>
#include <boost/algorithm/string/detail/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <vector>
#include <iosfwd>
#include <string>

namespace PVCore
{

class PVVersion
{
  public:
	PVVersion(size_t maj = 0, size_t min = 0, size_t rev = 0)
	    : _major(maj), _minor(min), _revision(rev)
	{
	}

	PVVersion(const char* version) : PVVersion()
	{
		std::vector<std::string> res;
		boost::algorithm::split(res, version, boost::is_any_of(".,-"));

		assert(res.size() <= 3);
		if (res.size() >= 1) {
			_major = std::stoull(res[0]);
		}
		if (res.size() >= 2) {
			_minor = std::stoull(res[1]);
		}
		if (res.size() >= 3) {
			_revision = std::stoull(res[2]);
		}
	}

  public:
	size_t major() const { return _major; }
	size_t minor() const { return _minor; }
	size_t revision() const { return _revision; }

  public:
	bool operator==(const PVVersion& rhs) const
	{
		return _major == rhs._major and _minor == rhs._minor and _revision == rhs._revision;
	}

	bool operator<(const PVVersion& rhs) const
	{
		return _major < rhs._major or (_major == rhs._major and _minor < rhs._minor) or
		       (_major == rhs._major and _minor == rhs._minor and _revision < rhs._revision);
	}

	bool operator<=(const PVVersion& rhs) const
	{
		return _major <= rhs._major or (_major == rhs._major and _minor <= rhs._minor) or
		       (_major == rhs._major and _minor == rhs._minor and _revision <= rhs._revision);
	}

	bool operator!=(const PVVersion& rhs) const { return not operator==(rhs); }

	bool operator>(const PVVersion& rhs) const { return not operator<=(rhs); }

	bool operator>=(const PVVersion& rhs) const { return not operator<(rhs); }

  private:
	friend std::ostream& operator<<(std::ostream& out, const PVVersion& v);

  private:
	size_t _major;
	size_t _minor;
	size_t _revision;
};

} // namespace PVCore

#endif // __PVKERNEL_CORE_PVVERSION_H__

/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2017
 */

#ifndef __PVKERNEL_CORE_PVVERSION_H__
#define __PVKERNEL_CORE_PVVERSION_H__

#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/classification.hpp>

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

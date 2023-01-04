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

#ifndef INENDI_PVMAPPINGPROPERTIES_H
#define INENDI_PVMAPPINGPROPERTIES_H

#include <inendi/PVMappingFilter.h> // for PVMappingFilter, etc

#include <pvkernel/rush/PVAxisFormat.h> // for PVAxisFormat

#include <pvkernel/core/PVArgument.h> // for PVArgumentList

#include <pvbase/types.h> // for PVCol

#include <pvcop/db/array.h> // for array

#include <cassert>   // for assert
#include <algorithm> // for move
#include <string>    // for string

namespace PVCore
{
class PVSerializeObject;
} // namespace PVCore
namespace PVRush
{
class PVFormat;
} // namespace PVRush

namespace Inendi
{

/**
 * \class PVMappingProperties
 *
 * \brief Stored functions and variables that can to be modified by those functions
 */
class PVMappingProperties
{
  public:
	PVMappingProperties(PVRush::PVFormat const& fmt, PVCol idx);
	explicit PVMappingProperties(PVRush::PVAxisFormat const& axis);
	PVMappingProperties(std::string mode, PVCore::PVArgumentList args);

  public:
	void set_mode(std::string const& mode);
	void set_args(PVCore::PVArgumentList const& args);
	PVCore::PVArgumentList const& get_args() const { return _args; }
	inline PVMappingFilter::p_type get_mapping_filter() const
	{
		assert(_mapping_filter);
		return _mapping_filter;
	}
	inline std::string const& get_mode() const { return _mode; }
	inline bool is_uptodate() const { return _is_uptodate; }

	void set_minmax(pvcop::db::array&& minmax) { _minmax = std::move(minmax); }
	pvcop::db::array const& get_minmax() const { return _minmax; }

	void set_uptodate() { _is_uptodate = true; }
	inline void invalidate() { _is_uptodate = false; }

  public:
	void serialize_write(PVCore::PVSerializeObject& so) const;
	static PVMappingProperties serialize_read(PVCore::PVSerializeObject& so);

  private:
	pvcop::db::array _minmax;
	std::string _mode;
	PVMappingFilter::p_type _mapping_filter;
	PVCore::PVArgumentList _args;
	bool _is_uptodate;
};
} // namespace Inendi

#endif /* INENDI_PVMAPPINGPROPERTIES_H */

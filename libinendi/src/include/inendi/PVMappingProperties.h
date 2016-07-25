/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVMAPPINGPROPERTIES_H
#define INENDI_PVMAPPINGPROPERTIES_H

#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/rush/PVFormat.h>

#include <inendi/PVAxis.h>
#include <inendi/PVMappingFilter.h>

namespace Inendi
{

class PVMapping;

/**
* \class PVMappingProperties
*
* \brief Stored functions and variables that can to be modified by those functions
*/
class PVMappingProperties
{
	friend class PVCore::PVSerializeObject;
	friend class PVMapping;

  public:
	PVMappingProperties(PVRush::PVFormat const& fmt, PVCol idx);
	PVMappingProperties(PVRush::PVAxisFormat const& axis, PVCol idx);
	PVMappingProperties(std::string const& mode, PVCore::PVArgumentList args, PVCol idx);

	// For serialization
	PVMappingProperties() { _index = 0; }

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

  public:
	bool operator==(const PVMappingProperties& org);

  protected:
	void serialize_write(PVCore::PVSerializeObject& so);
	static PVMappingProperties serialize_read(PVCore::PVSerializeObject& so,
	                                          Inendi::PVMapping const& parent);
	void set_uptodate() { _is_uptodate = true; }
	inline void invalidate() { _is_uptodate = false; }
	void set_default_args(PVRush::PVAxisFormat const& axis);

  private:
	pvcop::db::array _minmax;
	PVCol _index;
	std::string _mode;
	PVMappingFilter::p_type _mapping_filter;
	PVCore::PVArgumentList _args;
	std::string _mode;
	bool _is_uptodate;
};
}

#endif /* INENDI_PVMAPPINGPROPERTIES_H */

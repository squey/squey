/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVFILTER_PVMAPPINGFILTERSTRING_H
#define PVFILTER_PVMAPPINGFILTERSTRING_H

#include <inendi/PVMappingFilter.h>

namespace Inendi
{

/**
 * Compute string default mapping.
 *
 * This mapping split first on string len, then sort them on value.
 */
class PVMappingFilterString : public PVMappingFilter
{
  public:
	PVMappingFilterString(
	    PVCore::PVArgumentList const& args = PVMappingFilterString::default_args());

	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override;

  public:
	/**
	 * Setter for case_sensitif information.
	 */
	void set_args(PVCore::PVArgumentList const& args) override;

	/**
	 * MetaInformation for this plugin.
	 */
	QString get_human_name() const override { return "String"; }

	std::unordered_set<std::string> list_usable_type() const override
	{
		return {"ipv4",         "datetime",     "datetime_us",   "datetime_ms",
		        "number_float", "number_int32", "number_uint32", "string"};
	}

  private:
	bool _case_sensitive; //!< Whether we should care about case for mapping.

	CLASS_FILTER(PVMappingFilterString)
};
}

#endif

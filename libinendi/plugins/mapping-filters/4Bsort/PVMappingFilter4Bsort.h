/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVFILTER_PVMAPPINGFILTER4BSORT_H
#define PVFILTER_PVMAPPINGFILTER4BSORT_H

#include <inendi/PVMappingFilter.h>

namespace Inendi
{

/**
 *  mapping sorted on first 4 bytes.
 */
class PVMappingFilter4Bsort : public PVMappingFilter
{
  public:
	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override;

	/**
	 * Meta information from this plugin.
	 */
	QString get_human_name() const override { return "Pseudo-sort on the first 4 bytes"; }

	std::unordered_set<std::string> list_usable_type() const override
	{
		return {"ipv4", "time", "number_float", "number_int32", "number_uint32", "string"};
	}

  protected:
	CLASS_FILTER_NOPARAM(PVMappingFilter4Bsort)
};
}

#endif /* PVFILTER_PVMAPPINGFILTER4BSORT_H */

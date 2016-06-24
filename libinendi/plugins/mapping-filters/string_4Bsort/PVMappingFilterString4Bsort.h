/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVFILTER_PVMAPPINGFILTERSTRING4BSORT_H
#define PVFILTER_PVMAPPINGFILTERSTRING4BSORT_H

#include <inendi/PVMappingFilter.h>

namespace Inendi
{

/**
 * String mapping sorted on first 4 bytes.
 */
class PVMappingFilterString4Bsort : public PVMappingFilter
{
  public:
	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override;

	/**
	 * Meta information from this plugin.
	 */
	QString get_human_name() const { return QString("Pseudo-sort on the first 4 bytes"); }

  protected:
	CLASS_FILTER_NOPARAM(PVMappingFilterString4Bsort)
};
}

#endif /* PVFILTER_PVMAPPINGFILTERSTRING4BSORT_H */

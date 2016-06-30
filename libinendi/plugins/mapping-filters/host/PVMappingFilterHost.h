/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERHOST_H
#define PVFILTER_PVMAPPINGFILTERHOST_H

#include <inendi/PVMappingFilter.h>

namespace Inendi
{

/**
 * Signed integer mapping. It keeps integer values.
 */
class PVMappingFilterHost : public PVMappingFilter
{
  public:
	PVMappingFilterHost();

	/**
	 * Copy NRaw values (real integers value) as mapping value.
	 */
	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override;

	std::unordered_set<std::string> list_usable_type() const override
	{
		return {"ipv4", "time", "number_float", "number_int32", "number_uint32", "string"};
	}

	/**
	 * Metainformation for this plugin.
	 */
	QString get_human_name() const override { return QString("Host"); }

	CLASS_FILTER_NOPARAM(PVMappingFilterHost)
};
}

#endif

/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERDEFAULT_H

#include <inendi/PVMappingFilter.h>

#include <pvkernel/rush/PVNraw.h>

#include <boost/date_time/posix_time/posix_time.hpp>

namespace Inendi
{

class PVMappingFilterDefault;

/**
 * Signed integer mapping. It keeps integer values.
 */
class PVMappingFilterDefault : public PVMappingFilter
{
  public:
	PVMappingFilterDefault();

	/**
	 * Point mapping values directly to nraw values without any copy
	 */
	pvcop::db::array operator()(PVCol const /*col*/, PVRush::PVNraw const& /*nraw*/) override
	{
		return {};
	}

	std::unordered_set<std::string> list_usable_type() const override
	{
		return {"ipv4",          "ipv6",         "mac_address",   "time",
		        "duration",      "number_float", "number_double", "number_int64",
		        "number_uint64", "number_int32", "number_uint32", "number_int16",
		        "number_uint16", "number_int8",  "number_uint8",  "string"};
	}

	/**
	 * Metainformation for this plugin.
	 */
	QString get_human_name() const override { return QString("Default"); }

	CLASS_FILTER_NOPARAM(PVMappingFilterDefault)
};
}

#endif

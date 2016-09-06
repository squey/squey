/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#ifndef __INENDI_PVMAPPINGFILTER_UNIFORM_H__
#define __INENDI_PVMAPPINGFILTER_UNIFORM_H__

#include <inendi/PVMappingFilter.h>

namespace Inendi
{

class PVMappingFilterUniform : public PVMappingFilter
{
  public:
	PVMappingFilterUniform() { INIT_FILTER_NOPARAM(PVMappingFilterUniform); }

	pvcop::db::array operator()(PVCol const col, PVRush::PVNraw const& nraw) override;

	std::unordered_set<std::string> list_usable_type() const override
	{
		return {"ipv4",         "ipv6",         "mac_address",   "time",
		        "number_float", "number_int32", "number_uint32", "string"};
	}

	QString get_human_name() const override { return QString("Uniform"); }

	CLASS_FILTER_NOPARAM(PVMappingFilterUniform)
};

} // namespace inendi

#endif // __INENDI_PVMAPPINGFILTER_UNIFORM_H__

/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERHOSTDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERHOSTDEFAULT_H

#include <pvkernel/core/general.h>
#include <inendi/PVMappingFilter.h>

namespace Inendi {

struct host_mapping
{
	static Inendi::PVMappingFilter::decimal_storage_type process_utf8(const char* buf, size_t size, PVMappingFilter* m);
	static Inendi::PVMappingFilter::decimal_storage_type process_utf16(uint16_t const* buf, size_t size, PVMappingFilter* m);
};

class PVMappingFilterHostDefault: public PVMappingFilter
{
public:
	PVMappingFilterHostDefault(PVCore::PVArgumentList const& args = PVMappingFilterHostDefault::default_args());

public:
	decimal_storage_type* operator()(PVCol const col, PVRush::PVNraw const& nraw)
	{
		auto array = nraw.collection().column(col);
		for(size_t row=0; row< array.size(); row++) {
			std::string content = array.at(row);
			this->_dest[row] = host_mapping::process_utf8(content.c_str(), content.size(), this);
		}

		return this->_dest;
	}
	QString get_human_name() const override { return QString("Default"); }
	void set_args(PVCore::PVArgumentList const& args) override;
	PVCore::DecimalType get_decimal_type() const override { return PVCore::UnsignedIntegerType; }

private:
	bool _case_sensitive;
	
	CLASS_FILTER(PVMappingFilterHostDefault)
};

}

#endif

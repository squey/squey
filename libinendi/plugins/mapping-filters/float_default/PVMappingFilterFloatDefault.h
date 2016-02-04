/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERFLOAT_H
#define PVFILTER_PVMAPPINGFILTERFLOAT_H

#include <pvkernel/core/general.h>
#include <inendi/PVMappingFilter.h>

#include <tbb/enumerable_thread_specific.h>

namespace Inendi {

struct float_mapping
{
	static Inendi::PVMappingFilter::decimal_storage_type process_utf8(const char* buf, size_t size, PVMappingFilter* m);
	static Inendi::PVMappingFilter::decimal_storage_type process_utf16(uint16_t const* buf, size_t size, PVMappingFilter* m);
};

class PVMappingFilterFloatDefault: public PVMappingFilter
{
	friend class float_mapping;
public:
	decimal_storage_type* operator()(PVCol const col, PVRush::PVNraw const& nraw)
	{
		auto array = nraw.collection().column(col);
		for(size_t row=0; row< array.size(); row++) {
			std::string content = array.at(row);
			this->_dest[row] = float_mapping::process_utf8(content.c_str(), content.size(), this);
		}

		return this->_dest;
	}

	QString get_human_name() const { return QString("Default"); }
	PVCore::DecimalType get_decimal_type() const override { return PVCore::FloatType; }

protected:
	tbb::enumerable_thread_specific<QString>& th_qs() { return _th_qs; }

private:
	tbb::enumerable_thread_specific<QString> _th_qs;

	CLASS_FILTER(PVMappingFilterFloatDefault)
};

}

#endif

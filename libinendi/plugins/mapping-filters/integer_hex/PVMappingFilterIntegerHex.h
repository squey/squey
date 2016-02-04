/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERINTEGERHEX_H
#define PVFILTER_PVMAPPINGFILTERINTEGERHEX_H

#include <pvkernel/core/general.h>
#include <inendi/PVMappingFilter.h>

namespace Inendi {

class PVMappingFilterIntegerHex;

struct integer_mapping
{
	static Inendi::PVMappingFilter::decimal_storage_type process_utf8(const char* buf, size_t size, PVMappingFilter* m);
	static Inendi::PVMappingFilter::decimal_storage_type process_utf16(uint16_t const* buf, size_t size, PVMappingFilter* m);
};

class PVMappingFilterIntegerHex: public PVMappingFilter
{
	friend class integer_mapping;

public:
	PVMappingFilterIntegerHex(PVCore::PVArgumentList const& args = PVMappingFilterIntegerHex::default_args());

public:
	decimal_storage_type* operator()(PVCol const col, PVRush::PVNraw const& nraw)
	{
		auto array = nraw.collection().column(col);
		for(size_t row=0; row< array.size(); row++) {
			std::string content = array.at(row);
			this->_dest[row] = integer_mapping::process_utf8(content.c_str(), content.size(), this);
		}

		return this->_dest;
	}
	QString get_human_name() const override { return QString("Hexadecimal"); }
	PVCore::DecimalType get_decimal_type() const override;
	void set_args(PVCore::PVArgumentList const& args) override;

protected:
	inline bool is_signed() const { return _signed; }

private:
	bool _signed;

	CLASS_FILTER(PVMappingFilterIntegerHex)
};

}

#endif

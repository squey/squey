/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVFILTER_PVFIELDCONVERTERSUBSTITUTION_H
#define PVFILTER_PVFIELDCONVERTERSUBSTITUTION_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

#include <unordered_map>

namespace PVFilter {

class PVFieldConverterSubstitution : public PVFieldsConverter {

public:
	PVFieldConverterSubstitution(PVCore::PVArgumentList const& args = PVFieldConverterSubstitution::default_args());

public:
	void set_args(PVCore::PVArgumentList const& args) override;
	PVCore::PVField& one_to_one(PVCore::PVField& field) override;

private:
	std::string _default_value;
	bool    _use_default_value;
	char _sep_char;
	char _quote_char;
	std::unordered_map<std::string, std::string> _key;

	CLASS_FILTER(PVFilter::PVFieldConverterSubstitution)
};

}

#endif // PVFILTER_PVFIELDCONVERTERSUBSTITUTION_H

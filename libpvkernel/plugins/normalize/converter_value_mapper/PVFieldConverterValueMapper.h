/**
 * \file PVFieldConverterValueMapper.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef PVFILTER_PVFIELDCONVERTERVALUEMAPPER_H
#define PVFILTER_PVFIELDCONVERTERVALUEMAPPER_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter {

class PVFieldConverterValueMapper : public PVFieldsConverter {

public:
	PVFieldConverterValueMapper(PVCore::PVArgumentList const& args = PVFieldConverterValueMapper::default_args());

public:
	void set_args(PVCore::PVArgumentList const& args);
	PVCore::PVField& one_to_one(PVCore::PVField& field) override;

private:

	CLASS_FILTER(PVFilter::PVFieldConverterValueMapper)
};

}

#endif // PVFILTER_PVFIELDCONVERTERVALUEMAPPER_H

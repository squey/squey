/**
 * \file PVMappingFilterIntegerDefault.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVMappingFilterIntegerDefault.h"

Picviz::PVMappingFilterIntegerDefault::PVMappingFilterIntegerDefault(PVCore::PVArgumentList const& args):
	PVMappingFilter(),
	_signed(true) // This will be changed by set_args anyway
{
	INIT_FILTER(PVMappingFilterIntegerDefault, args);
}

DEFAULT_ARGS_FILTER(Picviz::PVMappingFilterIntegerDefault)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("signed", "Signed integers")].setValue<bool>(true);
	return args;
}

void Picviz::PVMappingFilterIntegerDefault::set_args(PVCore::PVArgumentList const& args)
{
	Picviz::PVMappingFilter::set_args(args);
	_signed = args["signed"].toBool();
}

Picviz::PVMappingFilter::decimal_storage_type* Picviz::PVMappingFilterIntegerDefault::operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values)
{
	assert(_dest);
	assert(values.size() >= _dest_size);

	const ssize_t size = values.size();
	const bool is_signed = _signed;
	
	if (is_signed) {
#pragma omp parallel
		{
			QString stmp;
			// Looks like this can be fine optimised with hand made SSE/AVX optimisation
#pragma omp parallel for
			for (ssize_t i = 0; i < size; i++) {
				values[i].get_qstr(stmp);
				_dest[i].storage_as_int() = stmp.toInt();
			}
		}
	}
	else {
#pragma omp parallel
		{
			QString stmp;
			// Looks like this can be fine optimised with hand made SSE/AVX optimisation
#pragma omp parallel for
			for (ssize_t i = 0; i < size; i++) {
				values[i].get_qstr(stmp);
				_dest[i].storage_as_uint() = stmp.toUInt();
			}
		}
	}

	return _dest;
}

PVCore::DecimalType Picviz::PVMappingFilterIntegerDefault::get_decimal_type() const
{
	return (_signed) ? PVCore::IntegerType : PVCore::UnsignedIntegerType;
}

IMPL_FILTER_NOPARAM(Picviz::PVMappingFilterIntegerDefault)

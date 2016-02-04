/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVMappingFilterEnumDefault.h"

Inendi::PVMappingFilterEnumDefault::PVMappingFilterEnumDefault(PVCore::PVArgumentList const& args):
	PVMappingFilter(),
	_case_sensitive(true), // This will be changed by set_args anyway
	_poscount(0)
{
	INIT_FILTER(PVMappingFilterEnumDefault, args);
}

DEFAULT_ARGS_FILTER(Inendi::PVMappingFilterEnumDefault)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("convert-lowercase", "Convert strings to lower case")].setValue<bool>(false);
	return args;
}

void Inendi::PVMappingFilterEnumDefault::set_args(PVCore::PVArgumentList const& args)
{
	Inendi::PVMappingFilter::set_args(args);
	_case_sensitive = !args.at("convert-lowercase").toBool();
}

uint32_t Inendi::PVMappingFilterEnumDefault::_enum_position_factorize(uint32_t v)
{
	// From http://graphics.stanford.edu/~seander/bithacks.html#ReverseByteWith64BitsDiv
	
	// swap odd and even bits
	v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
	// swap consecutive pairs
	v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
	// swap nibbles ... 
	v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
	// swap bytes
	v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
	// swap 2-byte long pairs
	v = ( v >> 16             ) | ( v               << 16);

	return v;
}

void Inendi::PVMappingFilterEnumDefault::init()
{
}

IMPL_FILTER(Inendi::PVMappingFilterEnumDefault)

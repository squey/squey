/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVSortingFunc.h>


Inendi::PVSortingFunc::PVSortingFunc(PVCore::PVArgumentList const& l):
	PVCore::PVFunctionArgs<f_type>(l),
	PVCore::PVRegistrableClass<PVSortingFunc>()
{
}

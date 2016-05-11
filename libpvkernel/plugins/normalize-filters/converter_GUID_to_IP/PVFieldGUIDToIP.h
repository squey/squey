/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVFIELGUIDTOIP_H
#define PVFILTER_PVFIELGUIDTOIP_H

#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter
{

class PVFieldGUIDToIP : public PVFieldsConverter
{

  public:
	PVFieldGUIDToIP(PVCore::PVArgumentList const& args = PVFieldGUIDToIP::default_args());

  public:
	void set_args(PVCore::PVArgumentList const& args);
	PVCore::PVField& one_to_one(PVCore::PVField& field) override;

  private:
	bool _ipv6;

	CLASS_FILTER(PVFilter::PVFieldGUIDToIP)
};
}

#endif // PVFILTER_PVFIELGUIDTOIP_H

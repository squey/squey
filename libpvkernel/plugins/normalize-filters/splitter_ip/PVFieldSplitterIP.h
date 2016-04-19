/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVFIELDSPLITTERIP_H
#define PVFILTER_PVFIELDSPLITTERIP_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVFilter {

/**
 * Split an IP in multiple field.
 *
 * For ipv4:
 * xxx.yyy.zzz.www
 *
 * with params = 1,3 => indexes = 2, 2
 * fields are : xxx.yyy and zzz.www
 *
 * For ipv6:
 * aaaa:zzzz:eeee:rrrr:tttt:yyyy:uuuu:iiii
 *
 * with params = 0, 5, 6 => indexes = 1, 5, 1
 * fields are : aaaa, zzzz:eeee:rrrr:tttt:yyyy and uuuu
 *
 * @note incomplete ipv6 result in empty fields.
 */
class PVFieldSplitterIP : public PVFieldsSplitter {

public:
	// Separator between quad information for params.
	static const QString sep;

public:
	PVFieldSplitterIP(PVCore::PVArgumentList const& args = PVFieldSplitterIP::default_args());

public:
	void set_args(PVCore::PVArgumentList const& args);
	PVCore::list_fields::size_type one_to_many(PVCore::list_fields &l, PVCore::list_fields::iterator it_ins, PVCore::PVField &field);

private:
	bool _ipv6; //!< Wether we split on ipv6 (or ipv4)
	std::vector<size_t> _indexes; //!< Elements to keep from previous position.

	CLASS_FILTER(PVFilter::PVFieldSplitterIP)
};

}

#endif // PVFILTER_PVFIELDSPLITTERIP_H

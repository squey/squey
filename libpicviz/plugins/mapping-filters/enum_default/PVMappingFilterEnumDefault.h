//! \file PVMappingFilterIPv4Default.h
//! $Id: PVMappingFilterIPv4Default.h 2492 2011-04-25 05:41:54Z psaade $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVMAPPINGFILTERENUMDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERENUMDEFAULT_H

#include <pvcore/general.h>
#include <picviz/PVMappingFilter.h>
#include <tbb/concurrent_unordered_map.h>
#include <QString>
#include <pvcore/stdint.h>

// Used by the concurrennt hash map below
size_t tbb_hasher(const QString& str);

namespace Picviz {

class LibExport PVMappingFilterEnumDefault: public PVMappingFilter
{
public:
	float* operator()(PVRush::PVNraw::nraw_table_line const& values);
	void init_from_first(QString const& value);

protected:
	typedef QHash<QString, int> hash_values;
protected:
	uint64_t _poscount;
	hash_values _enum_hash;

	CLASS_FILTER(PVMappingFilterEnumDefault)
};

}

#endif

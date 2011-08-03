//! \file PVMappingEnum10Default.h
//! $Id$
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVMAPPINGFILTERENUM10DEFAULT_H
#define PVFILTER_PVMAPPINGFILTERENUM10DEFAULT_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>
#include <tbb/concurrent_unordered_map.h>
#include <QString>

// Used by the concurrennt hash map below
size_t tbb_hasher(const QString& str);

namespace Picviz {

class LibExport PVMappingFilterEnum10Default: public PVMappingFilter
{
public:
	float operator()(QString const& str);
	void init_from_first(QString const& value);

protected:
	typedef tbb::concurrent_unordered_map<QString, int> hash_values;
protected:
	int _poscount;
	hash_values _enum_hash;

	CLASS_FILTER(PVMappingFilterEnum10Default)
};

}

#endif	/* PVFILTER_PVMAPPINGFILTERENUM10DEFAULT_H */

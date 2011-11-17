//! \file PVMappingFilterIPv4Default.h
//! $Id: PVMappingFilterIPv4Default.h 2492 2011-04-25 05:41:54Z psaade $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVMAPPINGFILTERENUMDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERENUMDEFAULT_H

#include <pvkernel/core/general.h>
#include <picviz/PVMappingFilter.h>
#include <tbb/concurrent_unordered_map.h>
#include <QString>
#include <pvkernel/core/stdint.h>

// Used by the concurrennt hash map below
size_t tbb_hasher(const QString& str);

namespace Picviz {

class PVMappingFilterEnumDefault: public PVMappingFilter
{
public:
	float* operator()(PVRush::PVNraw::nraw_table_line const& values);
	QString get_human_name() const { return QString("Default"); }

protected:
	typedef QHash<QString, QVariant> hash_values;
protected:
	uint64_t _poscount;

	CLASS_FILTER(PVMappingFilterEnumDefault)
};

}

#endif

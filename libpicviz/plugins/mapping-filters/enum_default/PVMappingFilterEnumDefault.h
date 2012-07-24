/**
 * \file PVMappingFilterEnumDefault.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERENUMDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERENUMDEFAULT_H

#include <pvkernel/core/PVUnicodeString.h>
#include <pvkernel/core/general.h>
#include <pvkernel/core/stdint.h>
#include <picviz/PVMappingFilter.h>

#include <QString>
#include <QMetaType>

#include <tbb/concurrent_unordered_map.h>

// Used by the concurrennt hash map below
size_t tbb_hasher(const QString& str);

namespace Picviz {

class PVMappingFilterEnumDefault: public PVMappingFilter
{
	// This needs to be public for Q_DECLARE_METATYPE
public:
	typedef QHash<PVCore::PVUnicodeString, QVariant> hash_values;
	typedef QHash<PVCore::PVUnicodeStringHashNoCase, QVariant> hash_nocase_values;

public:
	PVMappingFilterEnumDefault(PVCore::PVArgumentList const& args = PVMappingFilterEnumDefault::default_args());

public:
	void set_args(PVCore::PVArgumentList const& args);
	float* operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values);
	QString get_human_name() const { return QString("Default"); }

private:
	template <class HashType>
	float* process(PVRush::PVNraw::const_trans_nraw_table_line const& values)
	{
		float retval = 0;
		qlonglong position = 0;
		HashType enum_hash;
		if (_grp_value && _grp_value->isValid()) {
			PVLOG_DEBUG("(mapping-enum) using previous values for enumeration\n");
			enum_hash = _grp_value->value<HashType>();
		}
		_poscount = 0;

		for (size_t i = 0; i < values.size(); i++) {
			typename HashType::iterator it_v = enum_hash.find(values[i]);
			if (it_v != enum_hash.end()) {
				position = it_v.value().toLongLong();
				retval = _enum_position_factorize(position);
			} else {
				_poscount++;
				enum_hash[values[i]] = QVariant((qlonglong)_poscount);
				retval = _enum_position_factorize(_poscount);
			}
			_dest[i] = retval;
		}

		if (_grp_value) {
			_grp_value->setValue<HashType>(enum_hash);
		}

		return _dest;
	}

	static float _enum_position_factorize(qlonglong enumber);
protected:
	uint64_t _poscount;
	bool _case_sensitive;

	CLASS_FILTER(PVMappingFilterEnumDefault)
};

}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(Picviz::PVMappingFilterEnumDefault::hash_values)
Q_DECLARE_METATYPE(Picviz::PVMappingFilterEnumDefault::hash_nocase_values)

#endif

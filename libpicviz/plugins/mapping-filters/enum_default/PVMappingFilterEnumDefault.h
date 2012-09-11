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
	// AG: this needs to be public for Q_DECLARE_METATYPE
public:
	typedef QHash<PVCore::PVUnicodeString, QVariant> hash_values;
	typedef QHash<PVCore::PVUnicodeStringHashNoCase, QVariant> hash_nocase_values;

public:
	PVMappingFilterEnumDefault(PVCore::PVArgumentList const& args = PVMappingFilterEnumDefault::default_args());

public:
	void set_args(PVCore::PVArgumentList const& args) override;
	decimal_storage_type* operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values) override;
	QString get_human_name() const override { return QString("Default"); }
	PVCore::DecimalType get_decimal_type() const override { return PVCore::UnsignedIntegerType; }

private:
	template <class HashType>
	decimal_storage_type* process(PVRush::PVNraw::const_trans_nraw_table_line const& values)
	{
		uint32_t position = 0;
		HashType enum_hash;
		if (_grp_value && _grp_value->isValid()) {
			PVLOG_DEBUG("(mapping-enum) using previous values for enumeration\n");
			enum_hash = _grp_value->value<HashType>();
		}
		uint32_t poscount = 0;

		for (size_t i = 0; i < values.size(); i++) {
			uint32_t retval;
			typename HashType::iterator it_v = enum_hash.find(values[i]);
			if (it_v != enum_hash.end()) {
				position = it_v.value().toUInt();
				retval = _enum_position_factorize(position);
			} else {
				poscount++;
				enum_hash[values[i]] = QVariant(poscount);
				retval = _enum_position_factorize(poscount);
			}
			_dest[i].storage_as_uint() = retval;
		}

		if (_grp_value) {
			_grp_value->setValue<HashType>(enum_hash);
		}

		return _dest;
	}

private:
	static uint32_t _enum_position_factorize(uint32_t v);

protected:
	bool _case_sensitive;

	CLASS_FILTER(PVMappingFilterEnumDefault)
};

}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(Picviz::PVMappingFilterEnumDefault::hash_values)
Q_DECLARE_METATYPE(Picviz::PVMappingFilterEnumDefault::hash_nocase_values)

#endif

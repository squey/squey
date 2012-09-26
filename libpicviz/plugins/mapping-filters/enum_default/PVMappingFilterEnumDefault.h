/**
 * \file PVMappingFilterEnumDefault.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTERENUMDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERENUMDEFAULT_H

#include <pvkernel/core/PVUnicodeString.h>
#include <pvkernel/core/PVUnicodeString16.h>
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
	typedef QHash<QString, QVariant> hash_values;
	//typedef QHash<PVCore::PVUnicodeStringHashNoCase, QVariant> hash_nocase_values;
	//typedef QHash<PVCore::PVUnicodeString16, QVariant> hash16_values;
	//typedef QHash<PVCore::PVUnicodeString16HashNoCase, QVariant> hash16_nocase_values;

public:
	PVMappingFilterEnumDefault(PVCore::PVArgumentList const& args = PVMappingFilterEnumDefault::default_args());

public:
	void set_args(PVCore::PVArgumentList const& args) override;
	decimal_storage_type* operator()(PVCol const col, PVRush::PVNraw const& nraw) override;
	decimal_storage_type operator()(PVCore::PVField const& field) override;
	QString get_human_name() const override { return QString("Default"); }
	PVCore::DecimalType get_decimal_type() const override { return PVCore::UnsignedIntegerType; }

	void init() override;

private:
	template <class HashType>
	decimal_storage_type* process_nraw(PVCol const c, PVRush::PVNraw const& nraw)
	{
		HashType enum_hash;
		if (_grp_value && _grp_value->isValid()) {
			PVLOG_DEBUG("(mapping-enum) using previous values for enumeration\n");
			enum_hash = _grp_value->value<HashType>();
		}
		uint32_t poscount = 0;

		nraw.visit_column(c, [&](PVRow i, const char* buf, size_t size)
			{
				_dest[i].storage_as_uint() = this->process<HashType>(PVCore::PVUnicodeString((PVCore::PVUnicodeString::utf_char*) buf, size), enum_hash, poscount);
			});

		if (_grp_value) {
			_grp_value->setValue<HashType>(enum_hash);
		}

		return _dest;
	}

	template <class HashType, class UniStr>
	uint32_t process(UniStr const& uni_str, HashType& enum_hash, uint32_t& poscount)
	{
		uint32_t retval;
		QString scopy(uni_str.get_qstr_copy());
		typename HashType::iterator it_v = enum_hash.find(scopy);
		if (it_v != enum_hash.end()) {
			const uint32_t position = it_v.value().toUInt();
			retval = _enum_position_factorize(position);
		} else {
			poscount++;
			enum_hash[scopy] = QVariant(poscount);
			retval = _enum_position_factorize(poscount);
		}
		return retval;
	}

private:
	static uint32_t _enum_position_factorize(uint32_t v);

protected:
	bool _case_sensitive;
	hash_values _hash16_v;
	//hash16_nocase_values _hash16_nocase_v;
	uint32_t _poscount;

	CLASS_FILTER(PVMappingFilterEnumDefault)
};

}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(Picviz::PVMappingFilterEnumDefault::hash_values)
//Q_DECLARE_METATYPE(Picviz::PVMappingFilterEnumDefault::hash_nocase_values)

#endif

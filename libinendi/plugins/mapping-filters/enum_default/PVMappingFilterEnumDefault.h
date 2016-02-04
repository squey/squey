/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILTER_PVMAPPINGFILTERENUMDEFAULT_H
#define PVFILTER_PVMAPPINGFILTERENUMDEFAULT_H

#include <pvkernel/core/PVUnicodeString.h>
#include <pvkernel/core/general.h>
#include <inendi/PVMappingFilter.h>

#include <QString>
#include <QMetaType>

// Used by the concurrennt hash map below
size_t tbb_hasher(const QString& str);

namespace Inendi {

class PVMappingFilterEnumDefault: public PVMappingFilter
{
	// AG: this needs to be public for Q_DECLARE_METATYPE
public:
	typedef QHash<QString, QVariant> hash_values;

public:
	PVMappingFilterEnumDefault(PVCore::PVArgumentList const& args = PVMappingFilterEnumDefault::default_args());

public:
	void set_args(PVCore::PVArgumentList const& args) override;

Inendi::PVMappingFilter::decimal_storage_type process_cell(const char* buf, size_t size) override {
			Inendi::PVMappingFilter::decimal_storage_type ds;
			ds.storage_as_uint() = this->process(PVCore::PVUnicodeString((PVCore::PVUnicodeString::utf_char*) buf, size));
			return ds;
		}
	QString get_human_name() const override { return QString("Default"); }
	PVCore::DecimalType get_decimal_type() const override { return PVCore::UnsignedIntegerType; }

	void init() override;

private:
	template <class UniStr>
	uint32_t process(UniStr const& uni_str)
	{
		uint32_t retval;
		QString scopy(uni_str.get_qstr_copy());
		typename hash_values::iterator it_v = enum_hash.find(scopy);
		if (it_v != enum_hash.end()) {
			const uint32_t position = it_v.value().toUInt();
			retval = _enum_position_factorize(position);
		} else {
			_poscount++;
			enum_hash[scopy] = QVariant(_poscount);
			retval = _enum_position_factorize(_poscount);
		}
		return retval;
	}

private:
	static uint32_t _enum_position_factorize(uint32_t v);

protected:
	bool _case_sensitive;
	hash_values enum_hash;
	uint32_t _poscount;

	CLASS_FILTER(PVMappingFilterEnumDefault)
};

}

// WARNING : This declaration MUST BE outside namespace's scope
Q_DECLARE_METATYPE(Inendi::PVMappingFilterEnumDefault::hash_values)
//Q_DECLARE_METATYPE(Inendi::PVMappingFilterEnumDefault::hash_nocase_values)

#endif

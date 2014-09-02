/**
 * \file PVMappingFilterIPv4Uniform.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef PVFILTER_PVMAPPINGFILTERIPV4UNIFORM_H
#define PVFILTER_PVMAPPINGFILTERIPV4UNIFORM_H

#include <picviz/PVMappingFilter.h>

#include <QString>
#include <QMetaType>

namespace Picviz {

class PVMappingFilterIPv4Uniform: public PVMappingFilter
{
public:
	typedef QHash<uint32_t, uint32_t> values_t;

public:
	PVMappingFilterIPv4Uniform(PVCore::PVArgumentList const& args = PVMappingFilterIPv4Uniform::default_args());

public:
	void init() override;

	QString get_human_name() const override;
	PVCore::DecimalType get_decimal_type() const override;

	decimal_storage_type* operator()(PVCol const col, PVRush::PVNraw const& nraw) override;
	decimal_storage_type operator()(PVCore::PVField const& field) override;

	decimal_storage_type* finalize(PVCol const c,PVRush::PVNraw const& nraw) override;

protected:
	values_t _values;

	CLASS_FILTER(PVMappingFilterIPv4Uniform)
};

}

Q_DECLARE_METATYPE(Picviz::PVMappingFilterIPv4Uniform::values_t)

#endif // PVFILTER_PVMAPPINGFILTERIPV4UNIFORM_H

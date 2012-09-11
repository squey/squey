/**
 * \file PVMappingFilter.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVMAPPINGFILTER_H
#define PVFILTER_PVMAPPINGFILTER_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVDecimalStorage.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/rush/PVNraw.h>

#include <picviz/PVMapped_types.h>

#include <QString>
#include <QStringList>
#include <QVector>

namespace PVRush {
class PVFormat;
}

#ifdef WIN32
#ifdef picviz_EXPORTS
#define PVLibPicvizDecl __declspec(dllexport)
#else
#define PVLibPicvizDecl __declspec(dllimport)
#endif
#else
#define PVLibPicvizDecl
#endif

namespace Picviz {

class PVLibPicvizDecl PVMappingFilter: public PVFilter::PVFilterFunctionBase<float*, PVRush::PVNraw::nraw_table_line const&>, public PVCore::PVRegistrableClass<PVMappingFilter>
{
public:
	typedef Picviz::mapped_decimal_storage_type decimal_storage_type;
	typedef boost::shared_ptr<PVMappingFilter> p_type;
	typedef PVMappingFilter FilterT;

public:
	PVMappingFilter();
public:
	// Here we provide a default implementation which call operator()(QString const&) over an OpenMP-parallelised
	// for loop over values
	virtual decimal_storage_type* operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values);

	virtual decimal_storage_type operator()(QString const& value);
	virtual void init_from_first(QString const& value);

	void set_dest_array(PVRow size, decimal_storage_type *ptr);
	//void set_axis(Picviz::PVAxis const& axis);

	void set_group_value(PVCore::PVArgument& group) { _grp_value = &group; }

	virtual QString get_human_name() const;

	virtual PVCore::DecimalType get_decimal_type() const { return PVCore::FloatType; }

public:
	static QStringList list_types();
	static QStringList list_modes(QString const& type);
protected:
	PVRow _dest_size;
	decimal_storage_type* _dest;
	PVCore::PVArgument* _grp_value;
};

typedef PVMappingFilter::func_type PVMappingFilter_f;

}

#ifdef WIN32
LibPicvizDeclExplicitTempl PVCore::PVClassLibrary<Picviz::PVMappingFilter>;
LibPicvizDeclExplicitTempl PVCore::PVTag<Picviz::PVMappingFilter>;
#endif

#endif

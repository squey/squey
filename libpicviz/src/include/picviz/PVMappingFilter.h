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

class PVLibPicvizDecl PVMappingFilter: public PVFilter::PVFilterFunctionBase<Picviz::mapped_decimal_storage_type, PVCore::PVField const&>, public PVCore::PVRegistrableClass<PVMappingFilter>
{
public:
	typedef Picviz::mapped_decimal_storage_type decimal_storage_type;
	typedef std::shared_ptr<PVMappingFilter> p_type;
	typedef PVMappingFilter FilterT;

public:
	PVMappingFilter();
public:
	// Two interfaces: one that compute the mapped values from a PVCore::PVChunk, and one
	// that does it from an already existing nraw.
	virtual decimal_storage_type operator()(PVCore::PVField const& field) = 0;
	virtual decimal_storage_type* operator()(PVCol const col, PVRush::PVNraw const& nraw) = 0;

	/**
	 * provide a post processing step for mapping computation
	 *
	 * in some cases, the mapping may not be directly computed and need
	 * extra information about data. In those cases,
	 * ::operator()(PVCore::PVField const&) is used to prepare those extra
	 * informations and this method compute for real the resulting mapping.
	 *
	 * Thid is the default implementation which does nothing special.

	 * @param col the column number
	 * @param nraw a reference on the nraw
	 *
	 * @return a pointer on the filled decimal storage array
	 */
	virtual decimal_storage_type* finalize(PVCol const col, PVRush::PVNraw const& nraw)
	{
		(void)col;
		(void)nraw;
		return _dest;
	}

	/*
	// Here we provide a default implementation which call operator()(QString const&) over an OpenMP-parallelised
	// for loop over values
	virtual decimal_storage_type* operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values);
	virtual decimal_storage_type operator()(QString const& value);
	*/

	virtual void init();

	void set_dest_array(PVRow size, decimal_storage_type *ptr);
	void set_group_value(PVCore::PVArgument& group) { _grp_value = &group; }

	virtual bool is_pure() const { return false; }

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

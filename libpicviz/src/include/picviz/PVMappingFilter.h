//! \file PVMappingFilter.h
//! $Id: PVMappingFilter.h 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVMAPPINGFILTER_H
#define PVFILTER_PVMAPPINGFILTER_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/rush/PVNraw.h>
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
	typedef boost::shared_ptr<PVMappingFilter> p_type;
	typedef PVMappingFilter FilterT;

public:
	PVMappingFilter();
public:
	// Here we provide a default implementation which call operator()(QString const&) over an OpenMP-parallelised
	// for loop over values
	virtual float* operator()(PVRush::PVNraw::const_trans_nraw_table_line const& values);

	virtual float operator()(QString const& value);
	virtual void init_from_first(QString const& value);

	void set_dest_array(PVRow size, float *ptr);
	void set_format(PVCol current_col, PVRush::PVFormat& format);

	void set_group_value(PVCore::PVArgument& group) { _grp_value = &group; }

	virtual QString get_human_name() const;
public:
	static QStringList list_types();
	static QStringList list_modes(QString const& type);
protected:
	PVRow _dest_size;
	float* _dest;
	PVRush::PVFormat* _format;
	PVCol _cur_col;
	PVCore::PVArgument* _grp_value;
};

typedef PVMappingFilter::func_type PVMappingFilter_f;

}

#ifdef WIN32
LibPicvizDeclExplicitTempl PVCore::PVClassLibrary<Picviz::PVMappingFilter>;
LibPicvizDeclExplicitTempl PVCore::PVTag<Picviz::PVMappingFilter>;
#endif

#endif

//! \file PVMappingFilter.h
//! $Id: PVPlottingFilter.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVPLOTTINGFILTER_H
#define PVFILTER_PVPLOTTINGFILTER_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <picviz/PVMandatoryMappingFilter.h>

namespace Picviz {

class LibPicvizDecl PVPlottingFilter: public PVFilter::PVFilterFunctionBase<float*, float*>, public PVCore::PVRegistrableClass<PVPlottingFilter>
{
public:
	typedef boost::shared_ptr<PVPlottingFilter> p_type;
	typedef PVPlottingFilter FilterT;

public:
	PVPlottingFilter();

public:
	virtual float* operator()(float* value);
	virtual float operator()(float value);

	void set_dest_array(PVRow size, float* arr);
	void set_mapping_mode(QString const& mapping_mode);
	void set_mandatory_params(Picviz::mandatory_param_map const& params);

	virtual void init_expand(float /*min*/, float /*max*/) { };
	virtual float expand_plotted(float value) const { return value; }

	virtual QString get_human_name() const;

	virtual bool can_expand() const { return false; }
public:
	static QStringList list_modes(QString const& type, bool only_expandable = false);
	static QList<p_type> list_modes_lib(QString const& type, bool only_expandable);
	static QString mode_from_registered_name(QString const& rn);
protected:
	QString _mapping_mode;
	PVRow _dest_size;
	float* _dest;
	Picviz::mandatory_param_map const* _mandatory_params;
};

typedef PVPlottingFilter::func_type PVPlottingFilter_f;

}

#ifdef WIN32
LibPicvizDeclExplicitTempl PVCore::PVClassLibrary<Picviz::PVPlottingFilter>;
LibPicvizDeclExplicitTempl PVCore::PVTag<Picviz::PVPlottingFilter>;
#endif

#endif

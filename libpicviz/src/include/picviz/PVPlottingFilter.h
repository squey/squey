/**
 * \file PVPlottingFilter.h
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

#ifndef PVFILTER_PVPLOTTINGFILTER_H
#define PVFILTER_PVPLOTTINGFILTER_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/filter/PVFilterFunction.h>
#include <picviz/PVMandatoryMappingFilter.h>
#include <picviz/PVMapped_types.h>

namespace Picviz {

class LibPicvizDecl PVPlottingFilter: public PVFilter::PVFilterFunctionBase<uint32_t*, mapped_decimal_storage_type const*>, public PVCore::PVRegistrableClass<PVPlottingFilter>
{
public:
	typedef boost::shared_ptr<PVPlottingFilter> p_type;
	typedef PVPlottingFilter FilterT;

public:
	PVPlottingFilter();

public:
	virtual uint32_t* operator()(mapped_decimal_storage_type const* value);
	virtual uint32_t operator()(mapped_decimal_storage_type const value);

	void set_decimal_type(PVCore::DecimalType decimal_type) { _decimal_type = decimal_type; }
	void set_dest_array(PVRow size, uint32_t* arr);
	void set_mapping_mode(QString const& mapping_mode);
	void set_mandatory_params(Picviz::mandatory_param_map const& params);

	virtual void init_expand(uint32_t const /*min*/, uint32_t const /*max*/) { };
	virtual uint32_t expand_plotted(uint32_t const value) const { return value; }

	virtual QString get_human_name() const;

	virtual bool can_expand() const { return false; }
public:
	static QStringList list_modes(QString const& type, bool only_expandable = false);
	static QList<p_type> list_modes_lib(QString const& type, bool only_expandable);
	static QString mode_from_registered_name(QString const& rn);

protected:
	void copy_mapped_to_plotted(mapped_decimal_storage_type const* value);

protected:
	QString _mapping_mode;
	PVRow _dest_size;
	uint32_t* _dest;
	Picviz::mandatory_param_map const* _mandatory_params;
	PVCore::DecimalType _decimal_type;
};

typedef PVPlottingFilter::func_type PVPlottingFilter_f;

}

#ifdef WIN32
LibPicvizDeclExplicitTempl PVCore::PVClassLibrary<Picviz::PVPlottingFilter>;
LibPicvizDeclExplicitTempl PVCore::PVTag<Picviz::PVPlottingFilter>;
#endif

#endif

/**
 * \file PVXmlParamParserData.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

/* 
 * File:   PVXmlParamParserData.h
 * Author: rpernaudat
 *
 * Created on 12 mai 2011, 11:29
 */

#ifndef PVXMLPARAMPARSERDATA_H
#define	PVXMLPARAMPARSERDATA_H
#include <QString>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVCompList.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVRush {

class LibKernelDecl PVXmlParamParserData {

public:    
	PVXmlParamParserData()
	{
		// By default, the number of children is unknown.
		nchildren = 0;
	}
	int axis_id;
	PVFilter::PVFieldsFilterReg_p filter_lib;
	PVCore::PVArgumentList filter_args;
	size_t nchildren;
	PVFilter::filter_child_axes_tag_t children_axes_tag;

	bool operator==(PVXmlParamParserData const& other) const {
		return filter_lib->registered_id() == other.filter_lib->registered_id() && PVCore::comp_hash(filter_args, other.filter_args);
	}

	bool operator!=(PVXmlParamParserData const& other) const { return !(*this == other); }
};

}
#endif	/* PVXMLPARAMPARSERDATA_H */


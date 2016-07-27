/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

/*
 * File:   PVXmlParamParserData.h
 * Author: rpernaudat
 *
 * Created on 12 mai 2011, 11:29
 */

#ifndef PVXMLPARAMPARSERDATA_H
#define PVXMLPARAMPARSERDATA_H
#include <QString>

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVCompList.h>
#include <pvkernel/filter/PVFieldsFilter.h>

namespace PVRush
{

class PVXmlParamParserData
{

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
};
}
#endif /* PVXMLPARAMPARSERDATA_H */

/* 
 * File:   PVXmlParamParserData.h
 * Author: rpernaudat
 *
 * Created on 12 mai 2011, 11:29
 */

#ifndef PVXMLPARAMPARSERDATA_H
#define	PVXMLPARAMPARSERDATA_H
#include <QString>

#include <pvcore/general.h>
#include <pvcore/PVArgument.h>
#include <pvfilter/PVFieldsFilter.h>

namespace PVRush {

class LibRushDecl PVXmlParamParserData {

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
};
}
#endif	/* PVXMLPARAMPARSERDATA_H */


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

class LibCoreDecl PVXmlParamParserData {

public:    
	int axis_id;
	PVFilter::PVFieldsFilterReg_p filter_lib;
	PVCore::PVArgumentList filter_args;
};
}
#endif	/* PVXMLPARAMPARSERDATA_H */


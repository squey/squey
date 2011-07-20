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

namespace PVRush {

class LibCoreDecl PVXmlParamParserData {

public:    
	int axis_id;
	QString filter_type; // "splitter/filter"
	QString filter_plugin_name; // "regexp,etc.."
	PVCore::PVArgumentList filter_args;
};
}
#endif	/* PVXMLPARAMPARSERDATA_H */


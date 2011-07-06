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

namespace PVCore {

class LibExport PVXmlParamParserData {

public:    
	enum Type {
		filter,splitter,splitter_url,splitter_pcap,splitter_csv
	};
	int id;
	QString exp;
	Type type;
	bool grep_include;
	QChar csv_delimiter;
};
}
#endif	/* PVXMLPARAMPARSERDATA_H */


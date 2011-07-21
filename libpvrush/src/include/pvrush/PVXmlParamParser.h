/* 
 * File:   PVXmlParamParser.h
 * Author: rpernaudat
 *
 * Created on 12 mai 2011, 11:27
 */

#ifndef PVXMLPARAMPARSER_H
#define	PVXMLPARAMPARSER_H

#include <QList>
#include <QDomElement>
#include <QDomNodeList>
#include <QDomDocument>
#include <QDebug>
#include <QString>
#include <QStringList>
#include <QFile>
#include <QMap>
#include <QTextStream>
#include <QHash>

#include <pvcore/general.h>
#include <pvrush/PVXmlParamParserData.h>

namespace PVRush {

class LibRushDecl PVXmlParamParserException
{
public:
	virtual QString what() = 0;
};

class LibRushDecl PVXmlParamParserExceptionPluginNotFound: public PVXmlParamParserException
{
	public:
		PVXmlParamParserExceptionPluginNotFound(QString type, QString plugin_name);
		QString what();
	protected:
		QString _what;
};

class LibRushDecl PVXmlParamParser {
public:
	typedef QList<PVXmlParamParserData> list_params;
public:
	PVXmlParamParser(QString nameFile);
	virtual ~PVXmlParamParser();
	int setDom(QDomElement, int l=-1);
	QList<QHash<QString, QString> > const& getAxes()const;
	QList<PVXmlParamParserData> const& getFields()const;
	QHash<int, QStringList> const& getTimeFormat()const;
    unsigned int getVersion(){return format_version;}
    
private:
	QList<PVXmlParamParserData> fields;
	QList<QHash<QString, QString> > axes;
	QHash<int, QStringList> time_format;
    unsigned int format_version;

	int countChild(QDomElement);
	QString getNodeName(QDomElement);
	QString getNodeRegExp(QDomElement);
	QString getNodeType(QDomElement);
	QString getNodeTypeGrep(QDomElement node);
};

}

#endif	/* PVXMLPARAMPARSER_H */


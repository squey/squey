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

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVXmlParamParserData.h>

namespace PVRush {

class LibKernelDecl PVXmlParamParserException
{
public:
	virtual QString what() = 0;
};

class LibKernelDecl PVXmlParamParserExceptionPluginNotFound: public PVXmlParamParserException
{
	public:
		PVXmlParamParserExceptionPluginNotFound(QString type, QString plugin_name);
		QString what();
	protected:
		QString _what;
};

class LibKernelDecl PVXmlParamParser {
public:
	typedef QList<PVXmlParamParserData> list_params;
public:
	PVXmlParamParser(QString const& nameFile);
	PVXmlParamParser(QDomElement const& rootNode);
	virtual ~PVXmlParamParser();

public:
	int setDom(QDomElement const& node, int id = -1);
	QList<QHash<QString, QString> > const& getAxes()const;
	QList<PVXmlParamParserData> const& getFields()const;
	QHash<int, QStringList> const& getTimeFormat()const;
    unsigned int getVersion() { return format_version; }
	void dump_filters();
	void clearFiltersData();

private:
	void setVersionFromRootNode(QDomElement const& node);
	void pushFilter(const QDomElement& elt, int newId);
    
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


/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

/*
 * File:   PVXmlParamParser.h
 * Author: rpernaudat
 *
 * Created on 12 mai 2011, 11:27
 */

#ifndef PVXMLPARAMPARSER_H
#define PVXMLPARAMPARSER_H

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
#include <QVector>

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/rush/PVAxisFormat.h>
#include <pvkernel/rush/PVXmlParamParserData.h>
#include <pvkernel/rush/PVFormat_types.h>

namespace PVRush
{

class PVInvalidFile : public std::runtime_error
{
  public:
	using std::runtime_error::runtime_error;
};

class PVXmlParamParser
{
  public:
	typedef QList<PVXmlParamParserData> list_params;
	typedef std::vector<PVCol> axes_comb_t;
	using fields_mask_t = std::vector<bool>;

  public:
	PVXmlParamParser(QString const& nameFile);
	PVXmlParamParser(QDomElement const& rootNode);
	virtual ~PVXmlParamParser();

  public:
	int setDom(QDomElement const& node, int id = -1);
	QList<PVAxisFormat> const& getAxes() const;
	QList<PVXmlParamParserData> const& getFields() const;
	const fields_mask_t& getFieldsMask() const { return _fields_mask; }
	unsigned int getVersion() const { return format_version; }
	size_t get_first_line() const { return _first_line; }
	size_t get_line_count() const { return _line_count; }
	void dump_filters();
	void clearFiltersData();
	axes_comb_t const& getAxesCombination() const { return _axes_combination; }

  private:
	void pushFilter(const QDomElement& elt, int newId);
	void parseFromRootNode(QDomElement const& node);
	void setAxesCombinationFromRootNode(QDomElement const& node);
	void setAxesCombinationFromString(QString const& str);
	void setLinesRangeFromRootNode(QDomElement const& rootNode);
	static PVAxisFormat::node_args_t
	getMapPlotParameters(QDomElement& elt, QString const& tag, QString& mode);

  private:
	QList<PVXmlParamParserData> fields;
	QList<PVAxisFormat> _axes;
	unsigned int format_version;
	axes_comb_t _axes_combination;
	size_t _first_line;
	size_t _line_count;
	fields_mask_t _fields_mask;

	int countChild(QDomElement);
	QString getNodeName(QDomElement);
	QString getNodeRegExp(QDomElement);
	QString getNodeType(QDomElement);
	QString getNodeTypeGrep(QDomElement node);
};
}

#endif /* PVXMLPARAMPARSER_H */

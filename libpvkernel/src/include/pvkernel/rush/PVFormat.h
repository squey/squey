/*
 * $Id: PVFormat.h 3181 2011-06-21 07:15:22Z aguinet $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVRUSH_FORMAT_H
#define PVRUSH_FORMAT_H

#include <QDateTime>
#include <QDomElement>
#include <QString>
#include <QStringList>
#include <QByteArray>
#include <QMap>
#include <QHash>
#include <QList>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVElementFilter.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/rush/PVXmlParamParser.h>
#include <pvkernel/rush/PVAxisFormat.h>

#include <pvkernel/rush/PVFormat_types.h>

#include <boost/shared_ptr.hpp>

/**
 * \class PVRush::Format
 * \defgroup Format Input Formating
 * \brief Formating a log file
 * @{
 *
 * A format is used to know how to split the input file or buffer in columns. It is based on a XML description
 * Then is then used by the normalization part.
 *
 */

#define FORMAT_CUSTOM_NAME "custom"

namespace PVRush {

class PVFormatException
{
	public:
		virtual QString what() = 0;
};

/**
* This is the Format class
*/
class LibKernelDecl PVFormat {
public:
	typedef PVFormat_p p_type;

private:
	/* QString key_axis; */
	/* QString axis_color; */
	/* QString axis_titlecolor; */
	/* QString axis_group; */
	/* QString axis_name; */
	/* QString axis_type; */
	/* QString axis_mapping; */
	/* QString axis_plotting; */
	/* QString time_format_string; */
	/* QString decode_type; */

	QString format_name; // human readable name, displayed in a widget for instance
	QString full_path;

public:
	PVFormat();
	PVFormat(QString const& format_name_, QString const& full_path_);
	~PVFormat();

	/* Methods */
	void clear();
	void debug();
	bool populate_from_xml(QString filename, bool allowNoFilters = false);
	bool populate_from_xml(QDomElement const& rootNode, bool allowNoFilters = false);
	bool populate(bool allowNoFilters = false);
	
	PVFilter::PVChunkFilter_f create_tbb_filters();
	PVFilter::PVElementFilter_f create_tbb_filters_elt();

	static QHash<QString, PVRush::PVFormat> list_formats_in_dir(QString const& format_name_prefix, QString const& dir);

	QString const& get_format_name() const;
	QString const& get_full_path() const;

	void dump_elts(bool dump) { _dump_elts = dump; }

	list_axes_t const& get_axes() { return _axes; }


public:
	/* Attributes */

	QHash<int, QStringList> time_format;

	// List of filters to apply
	PVRush::PVXmlParamParser::list_params filters_params;

	unsigned int axes_count;	//!< It is equivalent to the number of axes except we add the decoded axes. This property must be used to know the number of axes, never count using axes.count()
	
	int time_format_axis_id;

protected:
	PVFilter::PVFieldsBaseFilter_f xmldata_to_filter(PVRush::PVXmlParamParserData const& fdata);
	bool populate_from_parser(PVXmlParamParser& xml_parser, bool allowNoFilters = false);

protected:
	list_axes_t _axes;

protected:
	// "Widget" arguments of the format, like:
	//  * use netflow (for PCAP)
	// They are editable by the user at the opening of a file/whatever
	PVCore::PVArgumentList _widget_args;

private:
	std::list<PVFilter::PVFieldsBaseFilter_p> _filters_container;
	bool _dump_elts;
};

};

/*@}*/

#endif	/* PVCORE_FORMAT_H */

/*
 * $Id: PVFormat.h 3181 2011-06-21 07:15:22Z aguinet $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVCORE_FORMAT_H
#define PVCORE_FORMAT_H

#include <QDateTime>
#include <QDomElement>
#include <QString>
#include <QStringList>
#include <QByteArray>
#include <QMap>
#include <QHash>
#include <QList>

#include <pvcore/general.h>
#include <pvrush/PVXmlParamParser.h>
#include <pvcore/PVArgument.h>
#include <pvfilter/PVChunkFilter.h>
#include <pvfilter/PVElementFilter.h>
#include <pvfilter/PVFieldsFilter.h>

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
	class LibRushDecl PVFormat {
		public:
			typedef QList<QHash<QString, QString> > list_axes;
			typedef boost::shared_ptr<PVFormat> p_type;

		private:
			QString key_axis;
			QString axis_color;
			QString axis_titlecolor;
			QString axis_group;
			QString axis_name;
			QString axis_type;
			QString axis_mapping;
			QString axis_plotting;
			QString time_format_string;
			QString decode_type;

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

			/* Attributes */

			QHash<int, QStringList> time_format;

			QList<QHash<QString, QString> > axes;
			/* QHash<int, QList<QHash<QString, QString> > > decode_axes;	//!< Store the decode axis position in the key and the value containes the list which contains the differents hashes to set the axes properties once decoded. This works *exactly* like the axes member, except we have an ID that is a virtual axes which is going to be replaced afterwards. It updates the property axes_count. */
			QHash<int, QString> axis_decoder;	//!< Which decoder should we use for the wanted position

			QList<int> axes_combination;
			
			// List of filters to apply
			PVRush::PVXmlParamParser::list_params filters_params;

			unsigned int axes_count;	//!< It is equivalent to the number of axes except we add the decoded axes. This property must be used to know the number of axes, never count using axes.count()
			
			int time_format_axis_id;

		protected:
			PVFilter::PVFieldsBaseFilter_f xmldata_to_filter(PVRush::PVXmlParamParserData const& fdata);
			bool populate_from_parser(PVXmlParamParser& xml_parser, bool allowNoFilters = false);

		protected:

		protected:
			// "Widget" arguments of the format, like:
			//  * use netflow (for PCAP)
			// They are editable by the user at the opening of a file/whatever
			PVCore::PVArgumentList _widget_args;

		private:
			std::list<PVFilter::PVFieldsBaseFilter_p> _filters_container;
			bool _dump_elts;
	};

	typedef QHash<QString, PVRush::PVFormat> hash_formats;
	typedef PVFormat::p_type PVFormat_p;

};

/*@}*/

#endif	/* PVCORE_FORMAT_H */

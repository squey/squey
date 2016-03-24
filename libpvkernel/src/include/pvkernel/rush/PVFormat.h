/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVElementFilter.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/rush/PVXmlParamParser.h>
#include <pvkernel/rush/PVAxisFormat.h>
#include <pvkernel/rush/PVFormat_types.h>

#include <pvcop/formatter_desc_list.h>

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

class PVFormatException : public std::runtime_error
{
	public:
		using std::runtime_error::runtime_error;
};

class PVFormatInvalid: public PVFormatException
{
public:
	using PVFormatException::PVFormatException;
	PVFormatInvalid() : PVFormatException("invalid format (no filters and/or axes)") {}
};

class PVFormatUnknownType: public PVFormatException
{
public:
	using PVFormatException::PVFormatException;
};

class PVFormatNoTimeMapping: public PVFormatException
{
public:
	using PVFormatException::PVFormatException;
};


/**
* This is the Format class
*/
class PVFormat {
	friend class PVCore::PVSerializeObject;
public:
	typedef PVFormat_p p_type;

public:
	class Comparaison
	{
		friend class PVFormat;
	public:
		bool same() const { return !_need_extract & !_mapping & !_plotting & !_other; }
		bool need_extract() const { return _need_extract; }
		bool different_mapping() const { return _mapping; }
		bool different_plotting() const { return _plotting; }
		bool different_other_axes_properties() const { return _other; }
	protected:
		bool _need_extract;
		bool _mapping;
		bool _plotting;
		bool _other;
	};

private:

	QString format_name; // human readable name, displayed in a widget for instance
	QString full_path;

public:
	PVFormat();
	PVFormat(QString const& format_name_, QString const& full_path_);
	~PVFormat();

	pvcop::formatter_desc_list get_storage_format() const;

	/* Methods */
	void debug();
	bool populate_from_xml(QString filename, bool forceOneAxis = false);
	bool populate_from_xml(QDomElement const& rootNode, bool forceOneAxis = false);
	bool populate(bool forceOneAxis = false);

	Comparaison comp(PVFormat const& original) const;
	
	PVFilter::PVChunkFilter_f create_tbb_filters_autodetect(float timeout, bool *cancellation = nullptr);
	PVFilter::PVChunkFilterByElt* create_tbb_filters();
	PVFilter::PVElementFilter_f create_tbb_filters_elt();

	static QHash<QString, PVRush::PVFormat> list_formats_in_dir(QString const& format_name_prefix, QString const& dir);

	QString const& get_format_name() const;
	QString const& get_full_path() const;

	bool exists() const;

	void dump_elts(bool dump) { _dump_elts = dump; }
	void restore_invalid_evts(bool restore) { _restore_inv_elts = restore; }


	list_axes_t const& get_axes() const { return _axes; }
	std::vector<PVCol> const& get_axes_comb() const { return _axes_comb; }

	size_t get_first_line() const { return _first_line; }
	size_t get_line_count() const { return _line_count; }

	// Remove any fields from the IR of the format and only
	// keeps fields.
	void only_keep_axes();

	static pvcop::formatter_desc get_datetime_formatter_desc(const std::string& tf);

public:
	/* Attributes */

	QHash<int, QStringList> time_format;

	// List of filters to apply
	PVRush::PVXmlParamParser::list_params filters_params;

	unsigned int axes_count;	//!< It is equivalent to the number of axes except we add the decoded axes. This property must be used to know the number of axes, never count using axes.count()

	int time_format_axis_id;

protected:
	PVFilter::PVFieldsBaseFilter_f xmldata_to_filter(PVRush::PVXmlParamParserData const& fdata);
	bool populate_from_parser(PVXmlParamParser& xml_parser, bool forceOneAxis = false);

protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);

protected:
	list_axes_t _axes;
	std::vector<PVCol> _axes_comb;
	size_t _first_line;
	size_t _line_count;

protected:
	// "Widget" arguments of the format, like:
	//  * use netflow (for PCAP)
	// They are editable by the user at the opening of a file/whatever
	PVCore::PVArgumentList _widget_args;

private:
	std::list<PVFilter::PVFieldsBaseFilter_p> _filters_container;
	bool _dump_elts;
	bool _already_pop;
	bool _original_was_serialized;
	bool _restore_inv_elts;
};

};

/*@}*/

#endif	/* PVCORE_FORMAT_H */

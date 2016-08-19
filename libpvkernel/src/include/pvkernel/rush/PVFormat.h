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

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVChunkFilterByEltCancellable.h>
#include <pvkernel/filter/PVElementFilter.h>
#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/rush/PVXmlParamParser.h>
#include <pvkernel/rush/PVFormat_types.h>

#include <pvcop/formatter_desc_list.h>

/**
 * \class PVRush::Format
 * \defgroup Format Input Formating
 * \brief Formating a log file
 * @{
 *
 * A format is used to know how to split the input file or buffer in columns. It is based on a XML
 *description
 * Then is then used by the normalization part.
 *
 */

#define FORMAT_CUSTOM_NAME "custom"

namespace PVRush
{

class PVFormatException : public std::runtime_error
{
  public:
	using std::runtime_error::runtime_error;
};

class PVFormatInvalid : public PVFormatException
{
  public:
	using PVFormatException::PVFormatException;
	PVFormatInvalid() : PVFormatException("invalid format (no filters and/or axes)") {}
};

class PVFormatUnknownType : public PVFormatException
{
  public:
	using PVFormatException::PVFormatException;
};

class PVFormatNoTimeMapping : public PVFormatException
{
  public:
	using PVFormatException::PVFormatException;
};

/**
* This is the Format class
*/
class PVFormat
{
  public:
	typedef PVFormat_p p_type;
	using fields_mask_t = PVXmlParamParser::fields_mask_t;

  public:
	PVFormat();
	PVFormat(QString const& format_name_, QString const& full_path_);
	PVFormat(QDomElement const& rootNode, bool forceOneAxis = false);

	pvcop::formatter_desc_list get_storage_format() const;

	/* Methods */
	void debug() const;

	PVFilter::PVChunkFilterByEltCancellable
	create_tbb_filters_autodetect(float timeout, bool* cancellation = nullptr);
	PVFilter::PVChunkFilterByElt create_tbb_filters();
	std::unique_ptr<PVFilter::PVElementFilter> create_tbb_filters_elt();

	static QHash<QString, PVRush::PVFormat> list_formats_in_dir(QString const& format_name_prefix,
	                                                            QString const& dir);

	QString const& get_format_name() const;
	QString const& get_full_path() const;

	bool exists() const;

	QList<PVAxisFormat> const& get_axes() const { return _axes; }
	std::vector<PVCol> const& get_axes_comb() const { return _axes_comb; }

	size_t get_first_line() const { return _first_line; }
	size_t get_line_count() const { return _line_count; }

	bool have_grep_filter() const { return _have_grep_filter; }

	static pvcop::formatter_desc get_datetime_formatter_desc(const std::string& tf);

  private:
	PVFilter::PVFieldsBaseFilter_p xmldata_to_filter(PVRush::PVXmlParamParserData const& fdata);

	bool populate(bool forceOneAxis = false);
	bool populate_from_parser(PVXmlParamParser& xml_parser, bool forceOneAxis = false);
	bool populate_from_xml(QDomElement const& rootNode, bool forceOneAxis = false);
	bool populate_from_xml(QString filename, bool forceOneAxis = false);

  public:
	void serialize_write(PVCore::PVSerializeObject& so);
	static PVFormat serialize_read(PVCore::PVSerializeObject& so);

  private:
	QString format_name; // human readable name, displayed in a widget for instance
	QString full_path;

	// List of filters to apply
	PVRush::PVXmlParamParser::list_params filters_params;
	fields_mask_t _fields_mask;

  protected:
	QList<PVAxisFormat> _axes;
	std::vector<PVCol> _axes_comb;
	size_t _first_line;
	size_t _line_count;

	bool _have_grep_filter;
};
};

/*@}*/

#endif /* PVCORE_FORMAT_H */

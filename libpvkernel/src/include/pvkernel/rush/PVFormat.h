/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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

#include <unordered_set>

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

class PVFormatInvalidTime : public PVFormatException
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
	explicit PVFormat(QDomElement const& rootNode);

	pvcop::formatter_desc_list get_storage_format() const;
	std::unordered_set<std::string> get_time_formats() const;

	/* Methods */
	void debug() const;

	PVFilter::PVChunkFilterByEltCancellable
	create_tbb_filters_autodetect(float timeout, bool* cancellation = nullptr);
	PVFilter::PVChunkFilterByElt create_tbb_filters() const;
	std::unique_ptr<PVFilter::PVElementFilter> create_tbb_filters_elt() const;

	static QHash<QString, PVRush::PVFormat> list_formats_in_dir(QString const& format_name_prefix,
	                                                            QString const& dir);

	void set_format_name(QString const& name);
	void set_full_path(QString const& full_path);
	QString const& get_format_name() const;
	QString const& get_full_path() const;

	bool is_valid() const;

	QList<PVAxisFormat> const& get_axes() const { return _axes; }
	std::vector<PVCol> const& get_axes_comb() const { return _axes_comb; }

	void insert_axis(const PVAxisFormat& axis, PVCombCol pos, bool after = true);
	void delete_axis(PVCol pos);

	size_t get_first_line() const { return _first_line; }
	void set_first_line(size_t first_line) { _first_line = first_line; }
	size_t get_line_count() const { return _line_count; }

	bool have_grep_filter() const { return _have_grep_filter; }

	PVFormat add_input_name_column() const;
	bool has_multi_inputs() const { return _has_multi_inputs; }

	static pvcop::formatter_desc get_datetime_formatter_desc(const std::string& tf);

	void set_python_script(const QString& python_script, bool as_path, bool disabled);
	QString get_python_script(bool& as_path, bool& disabled) const;

  private:
	PVFilter::PVFieldsBaseFilter_p
	xmldata_to_filter(PVRush::PVXmlParamParserData const& fdata) const;

	bool populate();
	bool populate_from_parser(PVXmlParamParser& xml_parser);
	bool populate_from_xml(QDomElement const& rootNode);
	bool populate_from_xml(QString filename);

  public:
	void serialize_write(PVCore::PVSerializeObject& so) const;
	static PVFormat serialize_read(PVCore::PVSerializeObject& so);

  private:
	QString _format_name; // human readable name, displayed in a widget for instance
	QString _full_path;

	// List of filters to apply
	PVRush::PVXmlParamParser::list_params filters_params;
	fields_mask_t _fields_mask;

  protected:
	QList<PVAxisFormat> _axes;
	std::vector<PVCol> _axes_comb;
	size_t _first_line;
	size_t _line_count;
	QString _python_script;
	bool _python_script_is_path;
	bool _python_script_disabled;

	QDomDocument _dom;

	mutable bool _have_grep_filter;
	mutable bool _has_multi_inputs = false;
};
} // namespace PVRush
;

/*@}*/

#endif /* PVCORE_FORMAT_H */

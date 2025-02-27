//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVUtils.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/rush/PVXmlParamParser.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVAxisFormat.h>
#include <pvkernel/rush/PVNormalizer.h>
#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVChunkFilterByEltCancellable.h>
#include <pvkernel/filter/PVElementFilterByAxes.h>
#include <pvkernel/filter/PVFieldsMappingFilter.h>
#include <pvkernel/filter/PVFieldFilterGrep.h>
#include <assert.h>
#include <qbytearray.h>
#include <qcontainerfwd.h>
#include <qdom.h>
#include <qiodevice.h>
#include <qlist.h>
#include <qstring.h>
#include <qstringlist.h>
#include <qtypeinfo.h>
#include <stdlib.h>
#include <QFile>
#include <QDir>
#include <QHashIterator>
#include <QFileInfo>
#include <QTextStream>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pvbase/types.h"
#include "pvcop/formatter_desc.h"
#include "pvcop/formatter_desc_list.h"
#include "pvkernel/core/PVArgument.h"
#include "pvkernel/core/PVLogger.h"
#include "pvkernel/core/PVOrderedMap.h"
#include "pvkernel/core/PVSerializeObject.h"
#include "pvkernel/filter/PVElementFilter.h"
#include "pvkernel/filter/PVFieldsFilter.h"
#include "pvkernel/rush/PVFormat_types.h"
#include "pvkernel/rush/PVXmlParamParserData.h"
#include "type_safe/strong_typedef.hpp"

static const std::unordered_set<std::string> SUPPORTED_TYPES = {
    "string",       "number_uint32", "number_int32",  "number_uint64",
    "number_int64", "number_uint16", "number_int16",  "number_uint8",
    "number_int8",  "number_float",  "number_double", "time",
    "duration",     "ipv4",          "ipv6",          "mac_address"};

PVRush::PVFormat::PVFormat() : _format_name(""), _full_path(""), _have_grep_filter(false) {}

PVRush::PVFormat::PVFormat(QString const& format_name, QString const& full_path)
    : _format_name(format_name), _full_path(full_path), _have_grep_filter(false)
{
	if (_format_name.isEmpty() && !_full_path.isEmpty()) {
		QFileInfo info(_full_path);
		QString basename = info.baseName();
		_format_name = basename;
	}
	populate();
}

PVRush::PVFormat::PVFormat(QDomElement const& root_node) : PVFormat()
{
	populate_from_xml(root_node);
}

PVRush::PVFormat PVRush::PVFormat::add_input_name_column() const
{
	PVRush::PVFormat new_format;

	PVRush::PVXmlParamParser xml_parser(_full_path, true);
	new_format.populate_from_parser(xml_parser);

	new_format._has_multi_inputs = true;

	return new_format;
}

/**
 * ICU : http://userguide.icu-project.org/formatparse/datetime
 * boost : http://www.boost.org/doc/libs/1_55_0/doc/html/date_time/date_time_io.html
 */
pvcop::formatter_desc PVRush::PVFormat::get_datetime_formatter_desc(const std::string& tf)
{
	static constexpr const char delimiter = '\'';

	std::string formatter;

	auto contains = [&](const std::string& str, const std::string& substr) {
		/**
		 * As the delimiter are displayed by repeating it twice instead of escaping it, the
		 * delimiter occurrences number in a well-formed string is necessarily even. Literal
		 * blocks can also be trivially skipped.
		 *
		 * Testing if substr exists also consists in a simple strcmp with explicit boundary
		 * checking.
		 */
		size_t str_pos = 0;

		const size_t str_size = str.size();
		const size_t substr_size = substr.size();

		/* The case of unclosed literal blocks is implictly included in the while test
		 */
		while (str_pos < str_size) {
			if (str[str_pos] == delimiter) {
				/* The literal block case.
				 */
				while ((str_pos < str_size) && (str[++str_pos] != delimiter)) {
				}
				++str_pos;
			} else {
				size_t substr_pos = 0;

				if (str[str_pos] == substr[substr_pos]) {
					/* The substr case.
					 */
					while ((str_pos < str_size) && (substr_pos < substr_size) &&
					       (str[++str_pos] == substr[++substr_pos])) {
					}
					if (substr_pos == substr_size) {
						/* substr has been entirely tested, have found it.
						 */
						return true;
					}
				}
				++str_pos;
			}
		}

		return false;
	};

	auto contains_one_of = [&](const std::string& tf, const std::vector<std::string>& tokens) {
		return std::any_of(tokens.cbegin(), tokens.cend(),
		                   [&](const std::string& token) { return contains(tf, token); });
	};

	/**
	 * The proper formatter is determined this way :
	 *
	 * 1. "datetime"    (libc)  : if no milliseconds and no timezone
	 * 2. "datetime_us" (boost) : if milliseconds but
	 *                            - no 2 digit year
	 *                            - no 12h format,
	 *                            - no timezone
	 *                            - no milliseconds not preceded by a dot
	 * 3. "datetime_ms" (ICU)   : in any other cases
	 */
	// rfc_timezone = X, XX, x, xx, Z, ZZ, ZZZ
	bool rfc_timezone = (contains_one_of(tf, {"X"}) && not contains_one_of(tf, {"XXX"})) ||
	                    (contains_one_of(tf, {"x"}) && not contains_one_of(tf, {"xxx"})) ||
	                    (contains_one_of(tf, {"Z"}) && not contains_one_of(tf, {"ZZZZ"}));
	bool no_timezone = not contains_one_of(tf, {"x", "X", "z", "Z", "v", "V"});
	bool no_extended_timezone = no_timezone || rfc_timezone;
	bool no_millisec_precision = not contains(tf, "S");
	bool no_epoch = not contains(tf, "epoch");
	bool no_12h_format = not contains(tf, "h") && no_epoch;
	bool no_two_digit_year = not(contains(tf, "yy") && not contains(tf, "yyyy"));

	if (no_millisec_precision && no_extended_timezone) {
		formatter = "datetime";
	} else {
		bool dot_before_millisec = contains(tf, ".S");

		if (dot_before_millisec && no_epoch && no_timezone && no_two_digit_year && no_12h_format) {
			formatter = "datetime_us";
		} else {
			// No need to make any format conversion as our input format is already good (ICU)
			return {"datetime_ms", tf};
		}
	}

	static std::vector<std::pair<std::string, std::string>> map = {// epoch
	                                                               {"epoch", "%s"},

	                                                               // year
	                                                               {"yyyy", "%Y"},
	                                                               {"yy", "%y"},

	                                                               // day of week
	                                                               {"eeee", "%a"},
	                                                               {"eee", "%a"},
	                                                               {"e", "%a"},
	                                                               {"EEEE", "%a"},
	                                                               {"EEE", "%a"},

	                                                               // month
	                                                               {"MMMM", "%b"},
	                                                               {"MMM", "%b"},
	                                                               {"MM", "%m"},
	                                                               {"M", "%m"},

	                                                               // day in month
	                                                               {"dd", "%d"},
	                                                               {"d", "%d"},

	                                                               // hour
	                                                               {"HH", "%H"},
	                                                               {"H", "%H"},
	                                                               {"hh", "%I"},
	                                                               {"h", "%I"},
	                                                               {"K", "%h"},

	                                                               // minute
	                                                               {"mm", "%M"},
	                                                               {"m", "%M"},

	                                                               // seconde
	                                                               {"ss", "%S"},
	                                                               {"s", "%S"},

	                                                               // fractional second
	                                                               {"SSSSSS", "%f"},
	                                                               {"SSSSS", "%f"},
	                                                               {"SSSS", "%f"},
	                                                               {"SSS", "%f"},
	                                                               {"SS", "%f"},
	                                                               {"S", "%f"},

	                                                               // am/pm marker
	                                                               {"aaa", "%p"},
	                                                               {"aa", "%p"},
	                                                               {"a", "%p"},

	                                                               // timezone
	                                                               {"Z", "%z"},
	                                                               {"zzzz", "%Z"},
	                                                               {"zzz", "%Z"},
	                                                               {"zz", "%Z"},
	                                                               {"z", "%Z"},
	                                                               {"v", "%Z"},
	                                                               {"VVV", "%Z"},
	                                                               {"V", "%z"}};

	std::string time_format = tf;

	// Handle litteral "%" that should be replaced by "%%" without interfering with the conversion
	PVCore::replace(time_format, "%", "☠");

	// iterate on the map with respect to the order of insertion
	for (const auto& token : map) {

		const std::string& key = token.first;
		const std::string& value = token.second;

		int pos = -value.size();
		while ((pos = time_format.find(key, pos + value.size())) != (int)std::string::npos) {

			// check that we are not in a '...' section
			bool verbatim =
			    std::count(time_format.begin(), time_format.begin() + pos, delimiter) % 2 == 1;
			if (not verbatim) {

				// Don't try to replace an already replaced token
				bool already_replaced_token = (pos > 0 && time_format[pos - 1] == '%');
				if (not already_replaced_token) {
					time_format.replace(pos, key.size(), value);
				}
			}
		}
	}

	PVCore::replace(time_format, "☠", "%%");

	// replace '' by ' and remove verbatim
	std::string value = "";
	std::string key = "'";
	int pos = -value.size();
	while ((pos = time_format.find(key, pos + value.size())) != (int)std::string::npos) {
		if (time_format[pos + 1] == '\'') {
			pos += 1;
			continue;
		}
		time_format.replace(pos, key.size(), value);
	}

	return {formatter, time_format};
}

std::unordered_set<std::string> PVRush::PVFormat::get_time_formats() const
{
	std::unordered_set<std::string> time_formats;

	for (const PVAxisFormat& axe : _axes) {
		if (axe.get_type() == "time") {
			time_formats.emplace(axe.get_type_format().toStdString());
		}
	}

	return time_formats;
}

pvcop::formatter_desc_list PVRush::PVFormat::get_storage_format() const
{
	pvcop::formatter_desc_list formatters;

	for (const PVAxisFormat& axe : _axes) {

		std::string axe_type = axe.get_type().toStdString();

		if (SUPPORTED_TYPES.find(axe_type) == SUPPORTED_TYPES.end()) {
			throw PVRush::PVFormatUnknownType("Unknown axis type : " + axe_type);
		}

		if (axe_type == "time") {
			std::string time_format = axe.get_type_format().toStdString();

			if (time_format.empty()) {
				throw PVFormatInvalidTime("No type format for axis '" +
				                          axe.get_name().toStdString() + "'");
			}

			formatters.emplace_back(get_datetime_formatter_desc(time_format));
		} else {
			std::string formatter = axe_type;
			std::string formatter_params = axe.get_type_format().toStdString();

			formatters.emplace_back(pvcop::formatter_desc(formatter, formatter_params));
		}
	}

	return formatters;
}

bool PVRush::PVFormat::populate()
{
	if (!_full_path.isEmpty()) {
		return populate_from_xml(_full_path);
	}

	throw std::runtime_error("We can't populate format without file");
}

void PVRush::PVFormat::set_format_name(QString const& name)
{
	_format_name = name;
}

void PVRush::PVFormat::set_full_path(QString const& full_path)
{
	_full_path = full_path;
	if (_format_name.isEmpty() && !_full_path.isEmpty()) {
		QFileInfo info(_full_path);
		QString basename = info.baseName();
		_format_name = basename;
	}
}

QString const& PVRush::PVFormat::get_format_name() const
{
	return _format_name;
}

QString const& PVRush::PVFormat::get_full_path() const
{
	return _full_path;
}

bool PVRush::PVFormat::is_valid() const
{
	return _axes.size() >= 2;
}

void PVRush::PVFormat::insert_axis(const PVAxisFormat& axis, PVCombCol /*pos*/, bool after /* = true */)
{
	_axes.append(axis);
	_axes_comb.emplace_back(PVCol(_axes.size()-1)); // TODO : don't insert at the end
	(void) after;
}

void PVRush::PVFormat::delete_axis(PVCol col)
{
	// Remove col from axes
	_axes.erase(std::remove_if(_axes.begin(), _axes.end(), [&](const PVAxisFormat& axis) { return axis.get_index() == col; }));
	for (PVAxisFormat& axis : _axes) {
		if (axis.get_index() > col) {
			axis.set_index(axis.get_index()-PVCol(1));
		}
	}

	// Remove col from axes combination (and update col values)
	_axes_comb.erase(std::remove(_axes_comb.begin(), _axes_comb.end(), col));
	for (PVCol& c : _axes_comb) {
		if (c > col) {
			c--;
		}
	}

	// Delete axis node from DOM
	QDomNodeList xml_axes = _dom.elementsByTagName(PVFORMAT_XML_TAG_AXIS_STR);
	QDomNode xml_axis = xml_axes.at(col);
   	xml_axis.parentNode().removeChild(xml_axis);

	// Update axes combination from DOM
	QDomNode axes_combination = _dom.documentElement().firstChildElement(PVFORMAT_XML_TAG_AXES_COMBINATION_STR);
	axes_combination.parentNode().removeChild(axes_combination);
	QStringList comb = axes_combination.toElement().text().split(",");
	comb.removeAll(QString::number(col.value()));

	for (QString& str : comb) {
		PVCol c(str.toUInt());
		if (c > col) {
			str = QString::number(c-1);
		}
	}
	QString new_axes_comb = comb.join(",");
	QDomElement ac = _dom.createElement(PVFORMAT_XML_TAG_AXES_COMBINATION_STR);
	QDomText new_axes_comb_text_node = _dom.createTextNode(new_axes_comb);
	ac.appendChild(new_axes_comb_text_node);
	_dom.documentElement().appendChild(ac);
}


char* fill_spaces(QString str, int max_spaces)
{
	// Use for debug so we display the different elements
	char* retbuf;

	retbuf = (char*)malloc(max_spaces + 1);

	int until = max_spaces - str.length();

	for (int i = 0; i < until; i++) {
		retbuf[i] = ' ';
		// retstr += " ";
	}

	retbuf[until] = '\0';

	return retbuf;
}

void PVRush::PVFormat::debug() const
{
	PVLOG_PLAIN("\n"
	            "id     |      type      |      mapping     |     scaling     |  color  |name \n");
	PVLOG_PLAIN(
	    "-------+----------------+------------------+------------------+---------+------...\n");

	unsigned int i = 0;
	for (const auto& axis : _axes) {
		char* fill;
		fill = fill_spaces(QString::number(i + 1, 10), 7);
		PVLOG_PLAIN("%d%s", i, fill);
		free(fill);
		fill = fill_spaces(axis.get_type(), 15);
		PVLOG_PLAIN("| %s%s", qPrintable(axis.get_type()), fill);
		free(fill);
		fill = fill_spaces(axis.get_mapping(), 17);
		PVLOG_PLAIN("| %s%s", qPrintable(axis.get_mapping()), fill);
		free(fill);
		fill = fill_spaces(axis.get_scaling(), 17);
		PVLOG_PLAIN("| %s%s", qPrintable(axis.get_scaling()), fill);
		free(fill);
		fill = fill_spaces(axis.get_color_str(), 8);
		PVLOG_PLAIN("| %s%s", qPrintable(axis.get_color_str()), fill);
		free(fill);
		PVLOG_PLAIN("| %s\n", qPrintable(axis.get_name()));
		i++;
	}

	// Dump filters
	if (filters_params.size() == 0) {
		PVLOG_PLAIN("No filters\n");
	} else {
		PVLOG_PLAIN("Filters:\n");
		PVLOG_PLAIN("--------\n");
		PVXmlParamParser::list_params::const_iterator it_filters;
		for (it_filters = filters_params.begin(); it_filters != filters_params.end();
		     it_filters++) {
			PVXmlParamParserData const& fdata = *it_filters;
			PVLOG_PLAIN("%d -> %s. Arguments:\n", fdata.axis_id,
			            qPrintable(fdata.filter_lib->registered_name()));
			PVCore::PVArgumentList const& args = fdata.filter_args;
			PVCore::PVArgumentList::const_iterator it_a;
			for (it_a = args.begin(); it_a != args.end(); it_a++) {
				PVLOG_PLAIN("'%s' = '%s'\n", qPrintable(it_a->key()),
				            qPrintable(PVCore::PVArgument_to_QString(it_a->value())));
			}
		}
	}
}

bool PVRush::PVFormat::populate_from_xml(QDomElement const& rootNode)
{
	// Keep a copy of the DOM document even if the format is stored on disk.
	// This will allow to serialize it with eventual columns deletion.
	_dom = rootNode.ownerDocument();

	PVRush::PVXmlParamParser xml_parser(rootNode);
	return populate_from_parser(xml_parser);
}

bool PVRush::PVFormat::populate_from_xml(QString filename)
{
	QFile file(filename);
	file.open(QIODevice::ReadOnly | QIODevice::Text);
	_dom.setContent(&file);

	PVRush::PVXmlParamParser xml_parser(filename);
	return populate_from_parser(xml_parser);
}

bool PVRush::PVFormat::populate_from_parser(PVXmlParamParser& xml_parser)
{
	filters_params = xml_parser.getFields();
	if (xml_parser.getAxes().size() == 0) {
		throw PVFormatInvalid("The format does not have any axis");
	}
	_axes = xml_parser.getAxes();
	_axes_comb = xml_parser.getAxesCombination();
	_fields_mask = xml_parser.getFieldsMask();
	_first_line = xml_parser.get_first_line();
	_line_count = xml_parser.get_line_count();
	_python_script = xml_parser.get_python_script(_python_script_is_path, _python_script_disabled);

	return true;
}

PVFilter::PVFieldsBaseFilter_p
PVRush::PVFormat::xmldata_to_filter(PVRush::PVXmlParamParserData const& fdata) const
{
	PVFilter::PVFieldsFilterReg_p filter_lib = fdata.filter_lib;
	assert(filter_lib);

	PVFilter::PVFieldsBaseFilter_p filter_clone = filter_lib->clone<PVFilter::PVFieldsBaseFilter>();

	// Check if this is a "one_to_many" filter, and, in such case, set the number of
	// expected fields.
	auto* sp_p =
	    dynamic_cast<PVFilter::PVFieldsFilter<PVFilter::one_to_many>*>(filter_clone.get());
	if (sp_p) {
		sp_p->set_number_expected_fields(fdata.nchildren);
	} else if (dynamic_cast<PVFilter::PVFieldFilterGrep*>(filter_clone.get())) {
		_have_grep_filter = true;
	}
	filter_clone->set_args(fdata.filter_args);

	return filter_clone;
}

PVFilter::PVChunkFilterByElt PVRush::PVFormat::create_tbb_filters() const
{
	return PVFilter::PVChunkFilterByElt{create_tbb_filters_elt()};
}

PVFilter::PVChunkFilterByEltCancellable
PVRush::PVFormat::create_tbb_filters_autodetect(float timeout, bool* cancellation)
{
	return PVFilter::PVChunkFilterByEltCancellable{create_tbb_filters_elt(), timeout, cancellation};
}

std::unique_ptr<PVFilter::PVElementFilter> PVRush::PVFormat::create_tbb_filters_elt() const
{
	PVLOG_INFO("Create filters for format %s\n", qPrintable(_format_name));

	std::unique_ptr<PVFilter::PVElementFilterByAxes> filter_by_axes(
	    new PVFilter::PVElementFilterByAxes(_fields_mask));

	// Here we create the pipeline according to the format
	for (PVRush::PVXmlParamParserData const& fdata : filters_params) {
		PVFilter::PVFieldsBaseFilter_p field_f = xmldata_to_filter(fdata);

		// Create the mapping (field_id)->field_filter
		filter_by_axes->add_filter(std::unique_ptr<PVFilter::PVFieldsBaseFilter>(
		    new PVFilter::PVFieldsMappingFilter(fdata.axis_id, field_f)));
	}

	// Finalise the pipeline
	return std::unique_ptr<PVFilter::PVElementFilter>(filter_by_axes.release());
}

QHash<QString, PVRush::PVFormat>
PVRush::PVFormat::list_formats_in_dir(QString const& format_name_prefix, QString const& dir)
{
	QHash<QString, PVRush::PVFormat> ret;
	QStringList normalize_helpers_dir_list = PVRush::normalize_get_helpers_plugins_dirs(dir);

	for (int counter = 0; counter < normalize_helpers_dir_list.count(); counter++) {
		QString normalize_helpers_dir_str(normalize_helpers_dir_list[counter]);
		PVLOG_INFO("Search for formats in %s\n", qPrintable(normalize_helpers_dir_str));
		QDir normalize_helpers_dir(normalize_helpers_dir_str);
		normalize_helpers_dir.setNameFilters(QStringList() << "*.format"
		                                                   << "*.pcre");
		QStringList files = normalize_helpers_dir.entryList();
		QStringListIterator filesIterator(files);
		while (filesIterator.hasNext()) {
			QString current_file = filesIterator.next();
			QFileInfo fileInfo(current_file);
			QString filename = fileInfo.completeBaseName();
			QString plugin_name = format_name_prefix + QString(":") + filename;
			PVLOG_INFO("Adding format '%s'\n", qPrintable(plugin_name));
			try {
				ret.insert(
				    plugin_name,
				    PVFormat(plugin_name, normalize_helpers_dir.absoluteFilePath(current_file)));
			} catch (PVRush::PVFormatInvalid const&) {
				PVLOG_INFO(("Format :" +
				            normalize_helpers_dir.absoluteFilePath(current_file).toStdString() +
				            " is invalid and can't be use")
				               .c_str());
				// If the format is invalid skip it
				continue;
			}
		}
	}

	return ret;
}

void PVRush::PVFormat::set_python_script(const QString& python_script, bool as_path, bool disabled)
{
	_python_script = python_script;
	_python_script_is_path = as_path;
	_python_script_disabled = disabled;
}

QString PVRush::PVFormat::get_python_script(bool& as_path, bool& disabled) const
{
	as_path = _python_script_is_path;
	disabled = _python_script_disabled;
	return _python_script;
}

PVRush::PVFormat PVRush::PVFormat::serialize_read(PVCore::PVSerializeObject& so)
{
	so.set_current_status("Loading format...");
	auto format_name = so.attribute_read<QString>("name");

	auto fname = so.attribute_read<QString>("filename");
	QString full_path = so.file_read(fname);
	QString pattern = PVRush::PVNrawCacheManager::nraw_dir() + "/investigation_tmp_XXXXXX";
	QString tmp_dir = PVCore::mkdtemp(pattern.toLatin1().data());
	QString new_full_path = tmp_dir + "/" + fname;
	std::rename(full_path.toStdString().c_str(), new_full_path.toStdString().c_str());

	return {format_name, new_full_path};
}

void PVRush::PVFormat::serialize_write(PVCore::PVSerializeObject& so) const
{
	const QString& format_name = "format";
	so.set_current_status("Saving format...");
	so.attribute_write("name", format_name);

	QString str;
	QTextStream stream(&str, QIODevice::WriteOnly);
	stream << _dom.toString();

	so.buffer_write("format", str.toLatin1(), str.size());
	so.attribute_write("filename", format_name);
}

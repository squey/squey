/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <QXmlStreamReader>
#include <QFile>
#include <QDir>
#include <QHashIterator>
#include <QDateTime>
#include <QFileInfo>

#include <pvkernel/core/PVFileSerialize.h>
#include <pvkernel/core/PVCompList.h>
#include <pvkernel/core/PVUtils.h>

#include <pvkernel/rush/PVXmlParamParser.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVAxisFormat.h>
#include <pvkernel/rush/PVNormalizer.h>

#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVChunkFilterByEltCancellable.h>
#include <pvkernel/filter/PVElementFilterByAxes.h>
#include <pvkernel/filter/PVFieldsMappingFilter.h>
#include <pvkernel/filter/PVFieldFilterGrep.h>

#include <pvcop/types/impl/formatter_factory.h>

PVRush::PVFormat::PVFormat() : _have_grep_filter(false)
{
}

PVRush::PVFormat::PVFormat(QString const& format_name_, QString const& full_path_) : PVFormat()
{
	full_path = full_path_;
	format_name = format_name_;

	if (format_name.isEmpty() && !full_path.isEmpty()) {
		QFileInfo info(full_path);
		QString basename = info.baseName();
		format_name = basename;
	}
	populate();
}

PVRush::PVFormat::PVFormat(QDomElement const& root_node, bool forceOneAxis)
    : format_name(""), full_path("")
{
	populate_from_xml(root_node, forceOneAxis);
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
	                                                               {"hh", "%l"},
	                                                               {"h", "%l"},
	                                                               {"K", "%h"},

	                                                               // minute
	                                                               {"mm", "%M"},
	                                                               {"m", "%M"},

	                                                               // seconde
	                                                               {"ss", "%S"},
	                                                               {"s", "%S"},

	                                                               // fractional second
	                                                               {"SSSSSS", "%F"},
	                                                               {"SSSSS", "%F"},
	                                                               {"SSSS", "%F"},
	                                                               {"SSS", "%F"},
	                                                               {"SS", "%F"},
	                                                               {"S", "%F"},

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

pvcop::formatter_desc_list PVRush::PVFormat::get_storage_format() const
{
	pvcop::formatter_desc_list formatters;

	for (const PVAxisFormat& axe : _axes) {

		std::string axe_type = axe.get_type().toStdString();

		if (axe_type == "time") {
			std::string time_format = axe.get_type_format().toStdString();

			if (time_format.empty()) {
				throw PVFormatNoTimeMapping(axe.get_name().toStdString());
			}

			formatters.emplace_back(get_datetime_formatter_desc(time_format));
		} else {
			std::string formatter;
			std::string formatter_params;

			if (axe_type == "string") {
				formatter = "string";
			} else if (axe_type == "number_uint32") {
				formatter = "number_uint32";
				formatter_params = axe.get_type_format().toStdString();
			} else if (axe_type == "number_int32") {
				formatter = "number_int32";
			} else if (axe_type == "number_float") {
				formatter = "number_float";
			} else if (axe_type == "ipv4") {
				formatter = "ipv4";
			} else {
				throw PVRush::PVFormatUnknownType("Unknown axis type : " + axe_type);
			}

			formatters.emplace_back(pvcop::formatter_desc(formatter, formatter_params));
		}
	}

	return formatters;
}

bool PVRush::PVFormat::populate(bool forceOneAxis)
{
	if (!full_path.isEmpty()) {
		return populate_from_xml(full_path, forceOneAxis);
	}

	throw std::runtime_error("We can't populate format without file");
}

QString const& PVRush::PVFormat::get_format_name() const
{
	return format_name;
}

QString const& PVRush::PVFormat::get_full_path() const
{
	return full_path;
}

bool PVRush::PVFormat::exists() const
{
	QFileInfo fi(full_path);

	return (fi.exists() && fi.isReadable());
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
	            "id     |      type      |      mapping     |     plotting     |  color  |name \n");
	PVLOG_PLAIN(
	    "-------+----------------+------------------+------------------+---------+------...\n");

	unsigned int i = 0;
	for (auto it = _axes.begin(); it != _axes.end(); it++) {
		char* fill;
		PVAxisFormat const& axis = *it;

		fill = fill_spaces(QString::number(i + 1, 10), 7);
		PVLOG_PLAIN("%d%s", i, fill);
		free(fill);
		fill = fill_spaces(axis.get_type(), 15);
		PVLOG_PLAIN("| %s%s", qPrintable(axis.get_type()), fill);
		free(fill);
		fill = fill_spaces(axis.get_mapping(), 17);
		PVLOG_PLAIN("| %s%s", qPrintable(axis.get_mapping()), fill);
		free(fill);
		fill = fill_spaces(axis.get_plotting(), 17);
		PVLOG_PLAIN("| %s%s", qPrintable(axis.get_plotting()), fill);
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

bool PVRush::PVFormat::populate_from_xml(QDomElement const& rootNode, bool forceOneAxis)
{
	PVRush::PVXmlParamParser xml_parser(rootNode);
	return populate_from_parser(xml_parser, forceOneAxis);
}

bool PVRush::PVFormat::populate_from_xml(QString filename, bool forceOneAxis)
{
	PVRush::PVXmlParamParser xml_parser(filename);
	return populate_from_parser(xml_parser, forceOneAxis);
}

bool PVRush::PVFormat::populate_from_parser(PVXmlParamParser& xml_parser, bool forceOneAxis)
{
	filters_params = xml_parser.getFields();
	if (filters_params.empty()) {
		throw PVFormatInvalid();
	}
	_axes = xml_parser.getAxes();
	_axes_comb = xml_parser.getAxesCombination();
	_fields_mask = xml_parser.getFieldsMask();
	_first_line = xml_parser.get_first_line();
	_line_count = xml_parser.get_line_count();

	if (_axes.size() == 0 && forceOneAxis) {
		// Only have one axis, a fake one
		PVAxisFormat fake_ax(-1);
		fake_ax.set_name("Line");
		fake_ax.set_type("string");
		fake_ax.set_mapping("default");
		fake_ax.set_plotting("default");
		fake_ax.set_color(PVFORMAT_AXIS_COLOR_DEFAULT);
		fake_ax.set_titlecolor(PVFORMAT_AXIS_TITLECOLOR_DEFAULT);
		_axes.clear();
		_axes.push_back(fake_ax);
		_fields_mask.resize(1, true);
	}

	return true;
}

PVFilter::PVFieldsBaseFilter_p
PVRush::PVFormat::xmldata_to_filter(PVRush::PVXmlParamParserData const& fdata)
{
	PVFilter::PVFieldsFilterReg_p filter_lib = fdata.filter_lib;
	assert(filter_lib);

	PVFilter::PVFieldsBaseFilter_p filter_clone = filter_lib->clone<PVFilter::PVFieldsBaseFilter>();

	// Check if this is a "one_to_many" filter, and, in such case, set the number of
	// expected fields.
	PVFilter::PVFieldsFilter<PVFilter::one_to_many>* sp_p =
	    dynamic_cast<PVFilter::PVFieldsFilter<PVFilter::one_to_many>*>(filter_clone.get());
	if (sp_p) {
		sp_p->set_number_expected_fields(fdata.nchildren);
	} else if (dynamic_cast<PVFilter::PVFieldFilterGrep*>(filter_clone.get())) {
		_have_grep_filter = true;
	}
	filter_clone->set_children_axes_tag(fdata.children_axes_tag, fdata.nchildren);
	filter_clone->set_args(fdata.filter_args);

	// initialize the filter
	filter_clone->init();

	return filter_clone;
}

PVFilter::PVChunkFilterByElt PVRush::PVFormat::create_tbb_filters()
{
	return PVFilter::PVChunkFilterByElt{create_tbb_filters_elt()};
}

PVFilter::PVChunkFilterByEltCancellable
PVRush::PVFormat::create_tbb_filters_autodetect(float timeout, bool* cancellation)
{
	return PVFilter::PVChunkFilterByEltCancellable{create_tbb_filters_elt(), timeout, cancellation};
}

std::unique_ptr<PVFilter::PVElementFilter> PVRush::PVFormat::create_tbb_filters_elt()
{
	PVLOG_INFO("Create filters for format %s\n", qPrintable(format_name));

	auto filter_by_axes = std::unique_ptr<PVFilter::PVElementFilterByAxes>(
	    new PVFilter::PVElementFilterByAxes(_fields_mask));

	// Here we create the pipeline according to the format
	for (PVRush::PVXmlParamParserData const& fdata : filters_params) {
		PVFilter::PVFieldsBaseFilter_p field_f = xmldata_to_filter(fdata);

		// Create the mapping (field_id)->field_filter
		PVFilter::PVFieldsBaseFilter_p mapping(
		    new PVFilter::PVFieldsMappingFilter(fdata.axis_id, field_f));
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

PVRush::PVFormat PVRush::PVFormat::serialize_read(PVCore::PVSerializeObject& so)
{
	so.set_current_status("Format loading");
	QString format_name;
	so.attribute("name", format_name);
	QString full_path;
	so.attribute("path", full_path);
	PVCore::PVFileSerialize format_file(full_path);
	if (so.object("file", format_file, "Include original format file", true,
	              (PVCore::PVFileSerialize*)nullptr, true, false)) {
		full_path = format_file.get_path();
	} else if (not QFileInfo(full_path).isReadable()) {
		if (so.is_repaired_error()) {
			full_path = QString::fromStdString(so.get_repaired_value());
		} else {
			throw PVCore::PVSerializeReparaibleError("Can't find format file",
			                                         so.get_logical_path().toStdString(),
			                                         full_path.toStdString());
		}
	}

	return {format_name, full_path};
}

void PVRush::PVFormat::serialize_write(PVCore::PVSerializeObject& so)
{
	so.set_current_status("Serialize format");
	so.attribute("name", format_name);
	so.attribute("path", full_path);
	PVCore::PVFileSerialize format_file(full_path);
	if (so.object("file", format_file, "Include original format file", true,
	              (PVCore::PVFileSerialize*)nullptr, true, false)) {
		full_path = format_file.get_path();
	}
}

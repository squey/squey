/*
 * $Id: PVFormat.cpp 3181 2011-06-21 07:15:22Z aguinet $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#include <QXmlStreamReader>
#include <QFile>
#include <QDir>
#include <QHashIterator>
#include <QDateTime>

#include <pvkernel/core/debug.h>
#include <pvkernel/core/PVFileSerialize.h>
#include <pvkernel/core/PVCompList.h>

#include <pvkernel/rush/PVXmlParamParser.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVNormalizer.h>

#include <pvkernel/filter/PVChunkFilterByElt.h>
#include <pvkernel/filter/PVChunkFilterByEltRestoreInvalid.h>
#include <pvkernel/filter/PVChunkFilterByEltSaveInvalid.h>
#include <pvkernel/filter/PVElementFilterByFields.h>
#include <pvkernel/filter/PVFieldSplitterUTF16Char.h>
#include <pvkernel/filter/PVFieldsMappingFilter.h>

PVRush::PVFormat::PVFormat()
{
	axes_count = 0;
	_dump_elts = false;
	_already_pop = false;
	_original_was_serialized = false;
	_restore_inv_elts = false;
}

PVRush::PVFormat::PVFormat(QString const& format_name_, QString const& full_path_)
{
	full_path = full_path_;
	format_name = format_name_;
	axes_count = 0;
	_dump_elts = false;
	_already_pop = false;
	_original_was_serialized = false;
}


PVRush::PVFormat::~PVFormat()
{
}



void PVRush::PVFormat::clear()
{

}

bool PVRush::PVFormat::populate(bool forceOneAxis)
{
	if (!full_path.isEmpty()) {
		return populate_from_xml(full_path, forceOneAxis);
	}

	return _already_pop;
}

QString const& PVRush::PVFormat::get_format_name() const
{
	return format_name;
}

QString const& PVRush::PVFormat::get_full_path() const
{
	return full_path;
}

char *fill_spaces(QString str, int max_spaces)
{
	// Use for debug so we display the different elements
	char *retbuf;

	retbuf = (char *)malloc(max_spaces + 1);

	int until = max_spaces - str.length();

	for (int i=0; i < until; i++) {
		retbuf[i] = ' ';
		// retstr += " ";
	}

	retbuf[until] = '\0';

	return retbuf;
}

void PVRush::PVFormat::debug()
{
	QHashIterator<int, QStringList> time_hash(time_format);

	PVLOG_PLAIN( "\nid     |      type      |      mapping     |     plotting     |    key    |    group    |  color  |name \n");
	PVLOG_PLAIN( "-------+----------------+------------------+------------------+-----------+-------------+---------+------...\n");

	list_axes_t::const_iterator it;
	unsigned int i = 0;
	for (it = _axes.begin(); it != _axes.end(); it++) {
		char *fill;
		PVAxisFormat const& axis = *it;

		fill = fill_spaces(QString::number(i+1, 10), 7);
		PVLOG_PLAIN( "%d%s", i, fill);
		free(fill);
		fill = fill_spaces(axis.get_type(), 15);
		PVLOG_PLAIN( "| %s%s", qPrintable(axis.get_type()), fill);
		free(fill);
		fill = fill_spaces(axis.get_mapping(), 17);
		PVLOG_PLAIN( "| %s%s", qPrintable(axis.get_mapping()), fill);
		free(fill);
		fill = fill_spaces(axis.get_plotting(), 17);
		PVLOG_PLAIN( "| %s%s", qPrintable(axis.get_plotting()), fill);
		free(fill);
		fill = fill_spaces(axis.get_group(), 12);
		PVLOG_PLAIN( "| %s%s", qPrintable(axis.get_group()), fill);
		free(fill);
		fill = fill_spaces(axis.get_color_str(), 8);
		PVLOG_PLAIN( "| %s%s", qPrintable(axis.get_color_str()), fill);
		free(fill);
		PVLOG_PLAIN( "| %s\n", qPrintable(axis.get_name()));
		i++;
	}

	// Dump filters
	if (filters_params.size() == 0) {
		PVLOG_PLAIN("No filters\n");
	}
	else {
		PVLOG_PLAIN("Filters:\n");
		PVLOG_PLAIN("--------\n");
		PVXmlParamParser::list_params::const_iterator it_filters;
		for (it_filters = filters_params.begin(); it_filters != filters_params.end(); it_filters++) {
			PVXmlParamParserData const& fdata = *it_filters;
			PVLOG_PLAIN("%d -> %s\n", fdata.axis_id, qPrintable(fdata.filter_lib->registered_name()));
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
	_axes = xml_parser.getAxes();
	_axes_comb = xml_parser.getAxesCombination();
	time_format = xml_parser.getTimeFormat();

	if (_axes.size() == 0 && forceOneAxis) {
		// Only have one axis, a fake one
		PVAxisFormat fake_ax;
		fake_ax.set_name("Line");
		fake_ax.set_type("string");
		fake_ax.set_mapping("default");
		fake_ax.set_plotting("default");
		fake_ax.set_group(PVFORMAT_AXIS_GROUP_DEFAULT);
		fake_ax.set_color(PVFORMAT_AXIS_COLOR_DEFAULT);
		fake_ax.set_titlecolor(PVFORMAT_AXIS_TITLECOLOR_DEFAULT);
		_axes.clear();
		_axes.push_back(fake_ax);
	}

	_already_pop = _axes.size() > 0;
	return _already_pop;
}

PVFilter::PVFieldsBaseFilter_f PVRush::PVFormat::xmldata_to_filter(PVRush::PVXmlParamParserData const& fdata)
{
	PVFilter::PVFieldsBaseFilter_f field_f;
	PVCore::PVArgumentList args;

	PVFilter::PVFieldsFilterReg_p filter_lib = fdata.filter_lib;
	assert(filter_lib);

	PVFilter::PVFieldsBaseFilter_p filter_clone = filter_lib->clone<PVFilter::PVFieldsBaseFilter>();
	// Check if this is a "one_to_many" filter, and, in such case, set the number of
	// expected fields.
	PVFilter::PVFieldsFilter<PVFilter::one_to_many>* sp_p = dynamic_cast<PVFilter::PVFieldsFilter<PVFilter::one_to_many>*>(filter_clone.get());
	if (sp_p) {
		sp_p->set_number_expected_fields(fdata.nchildren);
	}
	filter_clone->set_children_axes_tag(fdata.children_axes_tag);
	filter_clone->set_args(fdata.filter_args);
	_filters_container.push_back(filter_clone);
	field_f = filter_clone->f();

	return field_f;
}

PVFilter::PVChunkFilter_f PVRush::PVFormat::create_tbb_filters()
{
	PVFilter::PVElementFilter_f elt_f = create_tbb_filters_elt();
	assert(elt_f);
	PVFilter::PVChunkFilter* chk_flt;
	if (_dump_elts) {
		chk_flt = new PVFilter::PVChunkFilterByEltSaveInvalid(elt_f);
	}
	else
	if (_restore_inv_elts) {
		chk_flt = new PVFilter::PVChunkFilterByEltRestoreInvalid(elt_f);
	}
	else {
		chk_flt = new PVFilter::PVChunkFilterByElt(elt_f);
	}
	return chk_flt->f();
}

PVFilter::PVElementFilter_f PVRush::PVFormat::create_tbb_filters_elt()
{
	// We have to always return a valid filter function (even if this is
	// for a null processing filter).
	PVLOG_INFO("Create filters for format %s\n", qPrintable(format_name));
	if (filters_params.size() == 0) { // No filters, set an empty filter
		PVFilter::PVElementFilter* efnull = new PVFilter::PVElementFilter();
		return efnull->f();
	}

	PVFilter::PVFieldsBaseFilter_f first_filter = xmldata_to_filter(filters_params[0]);
	if (first_filter == NULL) {
		PVLOG_ERROR("Unknown first filter. Ignoring it !\n");
		PVFilter::PVElementFilter* efnull = new PVFilter::PVElementFilter();
		return efnull->f();
	}

	// Here we create the pipeline according to the format
	PVFilter::PVFieldsBaseFilter_f final_filter_f = first_filter;
	PVRush::PVXmlParamParser::list_params::const_iterator it_filters;
	if (filters_params.count() > 1) {
		for (it_filters = filters_params.begin()+1; it_filters != filters_params.end(); it_filters++) {
			PVRush::PVXmlParamParserData const& fdata = *it_filters;
			PVFilter::PVFieldsBaseFilter_f field_f = xmldata_to_filter(fdata);
			if (field_f == NULL) {
				PVLOG_ERROR("Unknown filter for field %d. Ignoring it !\n", fdata.axis_id);
				continue;
			}
			// Create the mapping (field_id)->field_filter
			PVFilter::PVFieldsMappingFilter::list_indexes indx;
			PVFilter::PVFieldsMappingFilter::map_filters mf;
			indx.push_back(fdata.axis_id);
			mf[indx] = field_f;
			PVFilter::PVFieldsBaseFilter_p mapping(new PVFilter::PVFieldsMappingFilter(mf));
			_filters_container.push_back(mapping);

			// Compose the pipeline
			final_filter_f = boost::bind(mapping->f(), boost::bind(final_filter_f, _1));
		}
	}

	// Finalise the pipeline
	PVFilter::PVElementFilterByFields* elt_f = new PVFilter::PVElementFilterByFields(final_filter_f);
	return elt_f->f();
}

QHash<QString, PVRush::PVFormat> PVRush::PVFormat::list_formats_in_dir(QString const& format_name_prefix, QString const& dir)
{
	QHash<QString, PVRush::PVFormat> ret;
	QStringList normalize_helpers_dir_list = PVRush::normalize_get_helpers_plugins_dirs(dir);

	for (int counter=0; counter < normalize_helpers_dir_list.count(); counter++) {
		QString normalize_helpers_dir_str(normalize_helpers_dir_list[counter]);
		if (normalize_helpers_dir_str.startsWith ("~/")) {
			normalize_helpers_dir_str.replace (0, 1, QDir::homePath());
		}

		PVLOG_INFO("Search for formats in %s\n", qPrintable(normalize_helpers_dir_str));	
		QDir normalize_helpers_dir(normalize_helpers_dir_str);
		normalize_helpers_dir.setNameFilters(QStringList() << "*.format" << "*.pcre");
		QStringList files = normalize_helpers_dir.entryList();
		QStringListIterator filesIterator(files);
		while (filesIterator.hasNext()) {
			QString current_file = filesIterator.next();
			QFileInfo fileInfo(current_file);
			QString filename = fileInfo.completeBaseName();
			QString plugin_name = format_name_prefix + QString(":") + filename;
			PVLOG_INFO("Adding format '%s'\n", qPrintable(plugin_name));
			ret.insert(plugin_name, PVFormat(plugin_name, normalize_helpers_dir.absoluteFilePath(current_file)));
		}
	}

	return ret;
}

void PVRush::PVFormat::only_keep_axes()
{
	// Remove the list of filters to apply, and only
	// keeps the fields !
	filters_params.clear();
	PVLOG_DEBUG("(PVRush::PVFormat) removing filters, we have '%d' fields.\n", _axes.size());
}

void PVRush::PVFormat::serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*v*/)
{
	so.attribute("name", format_name);
	so.attribute("path", full_path);
	PVCore::PVFileSerialize format_file(full_path);
	if (so.object("file", format_file, "Include original format file", !_original_was_serialized, (PVCore::PVFileSerialize*) NULL, !_original_was_serialized, false)) {
		full_path = format_file.get_path();
		if (!so.is_writing()) {
			_original_was_serialized = true;
		}
	}
	else
	if (!so.is_writing() && !QFileInfo(full_path).isReadable()) {
		boost::shared_ptr<PVCore::PVSerializeArchiveError> exc(new PVCore::PVSerializeArchiveErrorFileNotReadable(full_path));
		boost::shared_ptr<PVCore::PVSerializeArchiveFixAttribute> error(new PVCore::PVSerializeArchiveFixAttribute(so, exc, "path"));
		so.repairable_error(error);
		return;
	}
	populate();
}

PVRush::PVFormat::Comparaison PVRush::PVFormat::comp(PVFormat const& original) const
{
	Comparaison ret;
	ret._need_extract = (_axes.size() != original._axes.size()) ||
	                    (!PVCore::comp_list(filters_params, original.filters_params));
	if (ret._need_extract) {
		ret._mapping = true;
		ret._plotting = true;
		ret._other = true;
	}
	else
	if (_axes.size() == original._axes.size()) {
		list_axes_t::const_iterator it, it_org;
		it = _axes.begin();
		it_org = original._axes.begin();
		ret._mapping = false;
		ret._plotting = false;
		ret._other = false;
		for (; it != _axes.end(); it++) {
			if (it->get_type() != it_org->get_type()) {
				ret._mapping = true;
				ret._plotting = true;
				break;
			}
			if (it->get_mapping() != it_org->get_mapping() ||
				!PVCore::comp_hash(it->get_args_mapping(), it_org->get_args_mapping())) {
				ret._mapping = true;
			}
			if (it->get_plotting() != it_org->get_plotting() ||
				!PVCore::comp_hash(it->get_args_plotting(), it_org->get_args_plotting())) {
				ret._plotting = true;
			}
			if (it->get_tags() != it_org->get_tags() ||
				it->get_color_str() != it_org->get_color_str() ||
				it->get_titlecolor_str() != it_org->get_titlecolor_str() ||
				it->get_name() != it_org->get_name()) {
				ret._other = true;
			}
			it_org++;
		}
	}
	return ret;
}

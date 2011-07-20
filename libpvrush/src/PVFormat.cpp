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

#include <pvcore/debug.h>
#include <pvcore/PVXmlParamParser.h>

#include <pvrush/PVFormat.h>
#include <pvrush/PVNormalizer.h>

#include <pvfilter/PVChunkFilterByElt.h>
#include <pvfilter/PVElementFilterByFields.h>
#include <pvfilter/PVFieldSplitterUTF16Char.h>
#include <pvfilter/PVFieldsMappingFilter.h>

PVRush::PVFormat::PVFormat()
{
	axes_count = 0;
}

PVRush::PVFormat::PVFormat(QString const& format_name_, QString const& full_path_)
{
	full_path = full_path_;
	format_name = format_name_;
	axes_count = 0;
}


PVRush::PVFormat::~PVFormat()
{
}



void PVRush::PVFormat::clear()
{

}

bool PVRush::PVFormat::populate()
{
	return populate_from_xml(full_path);
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

	QHashIterator<int, QString> decoder_axis_hash(axis_decoder);

	while (decoder_axis_hash.hasNext()) {
		decoder_axis_hash.next();
		PVLOG_PLAIN("axis_decoder[%d]:'%s'\n", decoder_axis_hash.key(), decoder_axis_hash.value().toUtf8().data());
	}

	PVLOG_PLAIN( "\nid     |      type      |      mapping     |     plotting     |    key    |    group    |  color  |name \n");
	PVLOG_PLAIN( "-------+----------------+------------------+------------------+-----------+-------------+---------+------...\n");

	for (int i = 0; i < this->axes.size(); ++i) {
	  char *fill;

	  fill = fill_spaces(QString::number(i+1, 10), 7);
	  PVLOG_PLAIN( "%d%s", i+1, fill);
	  free(fill);
	  fill = fill_spaces(this->axes[i]["type"], 15);
	  PVLOG_PLAIN( "| %s%s", qPrintable(this->axes[i]["type"]), fill);
	  free(fill);
	  fill = fill_spaces(this->axes[i]["mapping"], 17);
	  PVLOG_PLAIN( "| %s%s", qPrintable(this->axes[i]["mapping"]), fill);
	  free(fill);
	  fill = fill_spaces(this->axes[i]["plotting"], 17);
	  PVLOG_PLAIN( "| %s%s", qPrintable(this->axes[i]["plotting"]), fill);
	  free(fill);
	  fill = fill_spaces(this->axes[i]["key"], 10);
	  PVLOG_PLAIN( "| %s%s", qPrintable(this->axes[i]["key"]), fill);
	  free(fill);
	  fill = fill_spaces(this->axes[i]["group"], 12);
	  PVLOG_PLAIN( "| %s%s", qPrintable(this->axes[i]["group"]), fill);
	  free(fill);
	  fill = fill_spaces(this->axes[i]["color"], 8);
	  PVLOG_PLAIN( "| %s%s", qPrintable(this->axes[i]["color"]), fill);
	  free(fill);
	  PVLOG_PLAIN( "| %s\n", qPrintable(this->axes[i]["name"]));

	}

}

bool PVRush::PVFormat::populate_from_xml(QString filename)
{
	PVCore::PVXmlParamParser xml_parser(filename);
	filters_params = xml_parser.getFields();
	axes = xml_parser.getAxes();
	time_format = xml_parser.getTimeFormat();

	return filters_params.size() > 0;
	//regex = filters_params[0].exp;
}

PVFilter::PVFieldsBaseFilter_f PVRush::PVFormat::xmldata_to_filter(PVCore::PVXmlParamParserData const& fdata)
{
	// TODO: this should be more generic than this !!
	PVFilter::PVFieldsSplitter::p_type regexp_lib_p = LIB_FILTER(PVFilter::PVFieldsSplitter)::get().get_filter_by_name("regexp");
	PVFilter::PVFieldsSplitter::p_type csv_lib_p = LIB_FILTER(PVFilter::PVFieldsSplitter)::get().get_filter_by_name("csv");
	PVFilter::PVFieldsSplitter::p_type url_lib_p = LIB_FILTER(PVFilter::PVFieldsSplitter)::get().get_filter_by_name("url");
	PVFilter::PVFieldsSplitter::p_type pcap_lib_p = LIB_FILTER(PVFilter::PVFieldsSplitter)::get().get_filter_by_name("pcap");
	PVFilter::PVFieldsFilter<PVFilter::one_to_one>::p_type grep_rx_lib_p = LIB_FILTER(PVFilter::PVFieldsFilter<PVFilter::one_to_one>)::get().get_filter_by_name("grep_regexp");

	PVFilter::PVFieldsBaseFilter_f field_f;
	PVCore::PVArgumentList args;

	if (fdata.type == PVCore::PVXmlParamParserData::splitter) {
		PVFilter::PVFieldsBaseFilter_p fre_in = regexp_lib_p->clone<PVFilter::PVFieldsBaseFilter>();
		_filters_container.push_back(fre_in);
		QRegExp re;
		re.setPattern(fdata.exp);
		args["regexp"] = re;
		fre_in->set_args(args);
		field_f = fre_in->f();

	}
	else
	if (fdata.type == PVCore::PVXmlParamParserData::splitter_url) {
		PVFilter::PVFieldsBaseFilter_p url = url_lib_p->clone<PVFilter::PVFieldsBaseFilter>();
		_filters_container.push_back(url);
		field_f = url->f();
	}
	else
	if (fdata.type == PVCore::PVXmlParamParserData::splitter_csv) {
		PVFilter::PVFieldsBaseFilter_p csv = csv_lib_p->clone<PVFilter::PVFieldsBaseFilter>();
		_filters_container.push_back(csv);
		args["sep"] = fdata.csv_delimiter;
		csv->set_args(args);
		field_f = csv->f();
	}
	else
	if (fdata.type == PVCore::PVXmlParamParserData::splitter_pcap) {
		PVLOG_INFO("pcap splitter found !\n");
		PVFilter::PVFieldsBaseFilter_p pcap = pcap_lib_p->clone<PVFilter::PVFieldsBaseFilter>();
		_filters_container.push_back(pcap);
		field_f = pcap->f();
	}
	else
	if (fdata.type == PVCore::PVXmlParamParserData::filter) {
		PVFilter::PVFieldsBaseFilter_p grep_rx = grep_rx_lib_p->clone< PVFilter::PVFieldsBaseFilter>();
		_filters_container.push_back(grep_rx);
		QRegExp re;
		re.setPattern(fdata.exp);
		args["regexp"] = re;
		args["reverse"] = !fdata.grep_include;
		grep_rx->set_args(args);
		field_f = grep_rx->f();
	}

	return field_f;

}

PVFilter::PVChunkFilter_f PVRush::PVFormat::create_tbb_filters()
{
	PVLOG_INFO("Create filters for format %s\n", qPrintable(format_name));
	if (filters_params.size() == 0) {
		PVLOG_ERROR("Invalid format: no filters !\n");
		return PVFilter::PVChunkFilter_f();
	}

	PVFilter::PVFieldsBaseFilter_f first_filter = xmldata_to_filter(filters_params[0]);
	if (first_filter == NULL) {
			PVLOG_ERROR("Unknown first filter. Ignoring it !\n");
			return PVFilter::PVChunkFilter_f();
	}

	// Here we create the pipeline according to the format
	PVFilter::PVFieldsBaseFilter_f final_filter_f = first_filter;
	PVCore::PVXmlParamParser::list_params::const_iterator it_filters;
	if (filters_params.count() > 1) {
		for (it_filters = filters_params.begin()+1; it_filters != filters_params.end(); it_filters++) {
			PVCore::PVXmlParamParserData const& fdata = *it_filters;
			PVFilter::PVFieldsBaseFilter_f field_f = xmldata_to_filter(fdata);
			if (field_f == NULL) {
				PVLOG_ERROR("Unknown filter for field %d. Ignoring it !\n", fdata.id);
				continue;
			}
			// Create the mapping (field_id)->field_filter
			PVFilter::PVFieldsMappingFilter::list_indexes indx;
			PVFilter::PVFieldsMappingFilter::map_filters mf;
			indx.push_back(fdata.id);
			mf[indx] = field_f;
			PVFilter::PVFieldsBaseFilter_p mapping(new PVFilter::PVFieldsMappingFilter(mf));
			_filters_container.push_back(mapping);

			// Compose the pipeline
			final_filter_f = boost::bind(mapping->f(), boost::bind(final_filter_f, _1));
		}
	}

	// Finalise the pipeline
	PVFilter::PVElementFilterByFields* elt_f = new PVFilter::PVElementFilterByFields(final_filter_f);
	PVFilter::PVChunkFilterByElt* chk_flt = new PVFilter::PVChunkFilterByElt(elt_f->f());

	return chk_flt->f();
}

QHash<QString, PVRush::PVFormat> PVRush::PVFormat::list_formats_in_dir(QString const& format_name_prefix, QString const& dir)
{
	QHash<QString, PVRush::PVFormat> ret;
	QStringList pcre_helpers_dir_list = PVRush::normalize_get_helpers_plugins_dirs(dir);
	PVLOG_INFO("Search for formats in %s\n", qPrintable(pcre_helpers_dir_list.join(" ")));

	for (int counter=0; counter < pcre_helpers_dir_list.count(); counter++) {
		QDir pcre_helpers_dir(pcre_helpers_dir_list[counter]);
		pcre_helpers_dir.setNameFilters(QStringList() << "*.format" << "*.pcre");
		QStringList files = pcre_helpers_dir.entryList();
		QStringListIterator filesIterator(files);
		while (filesIterator.hasNext()) {
			QString current_file = filesIterator.next();
			QFileInfo fileInfo(current_file);
			QString filename = fileInfo.completeBaseName();
			QString plugin_name = format_name_prefix + QString(":") + filename;
			ret.insert(plugin_name, PVFormat(plugin_name, pcre_helpers_dir.absoluteFilePath(current_file)));
		}
	}

	return ret;
}

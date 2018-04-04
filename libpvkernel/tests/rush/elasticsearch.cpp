/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#include "../../plugins/common/elasticsearch/PVElasticsearchAPI.h"
#include "../../plugins/common/elasticsearch/PVElasticsearchInfos.h"
#include "../../plugins/common/elasticsearch/PVElasticsearchQuery.h"

#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/rush/PVCSVExporter.h>
#include <pvkernel/core/PVUtils.h>

#include "common.h"

#include <fstream>

static const PVRush::PVElasticsearchAPI::columns_t
    ref_columns({{"@timestamp", {"time", "yyyy-MM-d'T'HH:mm:ss.S'Z'"}},
                 {"category", {"string", ""}},
                 {"geoip.latitude", {"number_float", ""}},
                 {"geoip.longitude", {"number_float", ""}},
                 {"http_method", {"string", ""}},
                 {"login", {"string", ""}},
                 {"mime_type", {"string", ""}},
                 {"src_ip", {"ipv6", ""}},
                 {"status_code", {"string", ""}},
                 {"time", {"time", "epochS"}},
                 {"time_spent", {"number_int32", ""}},
                 {"total_bytes", {"number_int32", ""}},
                 {"url", {"string", ""}},
                 {"user_agent", {"string", ""}}});
auto get_col_name = [](const auto& p) { return p.first; };

int main(int argc, char** argv)
{
	if (argc <= 1) {
		std::cerr << "Usage: " << argv[0] << " <query_export>" << std::endl;
		return 1;
	}

	pvtest::init_ctxt();

	/*
	 * Set Up an ElasticSearchInfo.
	 * It contains all information required to connect with the server
	 */
	PVRush::PVElasticsearchInfos infos;
	infos.set_host("http://connectors.srv.picviz");
	infos.set_port(9200);
	infos.set_index("proxy_sample_geoip");
	infos.set_login("elastic");
	infos.set_password("changeme");
	infos.set_filter_path(QString::fromStdString(
	    PVCore::join(boost::make_transform_iterator(ref_columns.begin(), get_col_name),
	                 boost::make_transform_iterator(ref_columns.end(), get_col_name), ",")));

	/*
	 * Set Up an ElasticSearchQuery.
	 * It contains all information required to define data to extract
	 */
	std::string query_str = R"###({
	  "condition": "AND",
	  "rules": [
		{
		  "id": "http_method",
		  "field": "http_method",
		  "type": "string",
		  "input": "text",
		  "operator": "equal",
		  "value": "get"
		},
		{
		  "id": "login",
		  "field": "login",
		  "type": "string",
		  "input": "text",
		  "operator": "not_equal",
		  "value": "11437"
		},
		{
		  "id": "login",
		  "field": "login",
		  "type": "string",
		  "input": "text",
		  "operator": "not_equal",
		  "value": "10715"
		},
		{
		  "condition": "OR",
		  "rules": [
			{
			  "id": "mime_type",
			  "field": "mime_type",
			  "type": "string",
			  "input": "text",
			  "operator": "not_equal",
			  "value": "image/jpeg"
			},
			{
			  "id": "time_spent",
			  "field": "time_spent",
			  "type": "integer",
			  "input": "text",
			  "operator": "greater",
			  "value": "10000"
			}
		  ]
		}
	  ],
	  "valid": true
	}
	)###";

	/*
	 *  Set Up the API from Information
	 */
	PVRush::PVElasticsearchAPI elasticsearch(infos);

	PVRush::PVElasticsearchQuery* query = new PVRush::PVElasticsearchQuery(
	    infos, QString(elasticsearch.rules_to_json(query_str.c_str()).c_str()), "json");

	std::string error;

	/**************************************************************************
	 * Check connection is correctly done with connection information
	 * TODO : Add a check with incorrect information?
	 *************************************************************************/
	bool connection_ok = elasticsearch.check_connection(&error);
	if (not error.empty()) {
		std::cout << error << std::endl;
	}
	PV_ASSERT_VALID(connection_ok);

	/**************************************************************************
	 * Check all indexes are available
	 * TODO : Check with no indexes and/or more than one?
	 *************************************************************************/
	PVRush::PVElasticsearchAPI::indexes_t indexes = elasticsearch.indexes(&error);
	if (not error.empty()) {
		std::cout << error << std::endl;
	}
	PV_ASSERT_VALID(std::find(indexes.begin(), indexes.end(), "proxy_sample") != indexes.end());

	/**************************************************************************
	 * Check all columns are available
	 * TODO : Check with no column?
	 *************************************************************************/
	PVRush::PVElasticsearchAPI::columns_t columns =
	    elasticsearch.format_columns(infos.get_filter_path().toStdString(), &error);
	if (not error.empty()) {
		std::cout << error << std::endl;
	}

	PV_ASSERT_VALID(columns == ref_columns);

	/**************************************************************************
	 * Check query count is correct
	 * TODO : What happen with no result?
	 *************************************************************************/
	size_t count = elasticsearch.count(*query, &error);
	if (not error.empty()) {
		std::cout << error << std::endl;
	}
	PV_VALID(count, 9981UL);

	/**************************************************************************
	 * Import data
	 *************************************************************************/
	std::string ref_file = argv[1];
	PVRush::PVSourceCreator_p sc =
	    LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("elasticsearch");
	QList<std::shared_ptr<PVRush::PVInputDescription>> list_inputs;
	list_inputs << PVRush::PVInputDescription_p(query);
	PVRush::PVNraw nraw;
	PVRush::PVNrawOutput output(nraw);
	PVRush::PVFormat format(elasticsearch.get_format_from_mapping().documentElement());
	PVRush::PVExtractor extractor(format, output, sc, list_inputs);

	// Import data
	auto start = std::chrono::system_clock::now();
	PVRush::PVControllerJob_p job =
	    extractor.process_from_agg_idxes(0, IMPORT_PIPELINE_ROW_COUNT_LIMIT);
	job->wait_end();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();

	// Export selected lines
	PVCore::PVSelBitField sel(nraw.row_count());
	sel.select_all();
	std::string output_file = pvtest::get_tmp_filename();
	PVRush::PVCSVExporter::export_func_f export_func =
	    [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
	        const std::string& quote) { return nraw.export_line(row, cols, sep, quote); };
	PVRush::PVCSVExporter exp(format.get_axes_comb(), nraw.row_count(), export_func);

	exp.export_rows(output_file, sel);

#ifndef INSPECTOR_BENCH
	// Check output is the same as the reference
	std::cout << std::endl << output_file << " - " << ref_file << std::endl;
	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(output_file, ref_file));
#endif

	std::remove(output_file.c_str());

	return 0;
}

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

#include <fstream>

int main(int argc, char** argv)
{
	if (argc <= 1) {
		std::cerr << "Usage: " << argv[0] << " <query_export>" << std::endl;
		return 1;
	}

	/*
	 * Set Up an ElasticSearchInfo.
	 * It contains all information required to connect with the server
	 */
	PVRush::PVElasticsearchInfos infos;
	infos.set_host("http://connectors.srv.picviz");
	infos.set_port(9200);
	infos.set_index("proxy_sample");

	/*
	 * Set Up an ElasticSearchQuery.
	 * It contains all information required to define data to extract
	 */
	std::string query_type = "json";
	std::string query_str =
		"{ \"query\" : { \"range\" : {"
			"\"total_bytes\" : {"
				"\"gte\": 350000"
			"}"
		 "} } }";
	PVRush::PVElasticsearchQuery query(infos, query_str.c_str(), query_type.c_str());

	/*
	 *  Set Up the API from Information
	 */
	PVRush::PVElasticsearchAPI elasticsearch(infos);

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
	PV_ASSERT_VALID(
		indexes.size() == 1 &&
		indexes[0] == "proxy_sample"
	);

	/**************************************************************************
	 * Check all columns are available
	 * TODO : Check with no column?
	 *************************************************************************/
	PVRush::PVElasticsearchAPI::columns_t columns = elasticsearch.columns(query, &error);
	if (not error.empty()) {
		std::cout << error << std::endl;
	}
	PV_ASSERT_VALID(columns == PVRush::PVElasticsearchAPI::columns_t({
		{"category", "string"},
		{"column13", "string"},
		{"http_method", "string"},
		{"login", "string"},
		{"mime_type", "string"},
		{"semi_colon", "string"},
		{"src_ip", "string"},
		{"status_code", "string"},
		{"time", "string"},
		{"time_spent", "integer"},
		{"total_bytes", "integer"},
		{"url", "string"},
		{"user_agent", "string"}
	 }));

	/**************************************************************************
	 * Check query count is correct
	 * TODO : What happen with no result?
	 *************************************************************************/
	size_t count = elasticsearch.count(query, &error);
	if (not error.empty()) {
		std::cout << error << std::endl;
	}

	PV_ASSERT_VALID(count == 10439);

	/**************************************************************************
	 * Check export query is correct
	 *
	 * @note files are sorted before comparison because elasticsearch doesn't
	 * return values in the same orders each times
	 *
	 * TODO : What happen with no result?
	 *************************************************************************/
	std::string output_file = "/tmp/Trush_elasticsearch.export";
	std::string reference_file = argv[1];
	std::string reference_sorted_file = output_file + "_sorted";
	std::ofstream output_stream(output_file, std::ios::out | std::ios::trunc);
	PVRush::PVElasticsearchAPI::rows_chunk_t rows_array;

	// Extract datas
	bool query_end = false;
	do {
		query_end = elasticsearch.extract(query, rows_array, &error);
		for (const PVRush::PVElasticsearchAPI::rows_t& rows : rows_array) {
			for (const std::string& row : rows) {
				output_stream << row.c_str() << std::endl;
			}
		}
	} while (query_end == false);

	// Count line in reference file
	std::ifstream reference_file_stream(reference_file);
	if(not reference_file_stream.good()) {
		return 1;
	}
	size_t reference_file_line_count = std::count(
		std::istreambuf_iterator<char>(reference_file_stream),
		std::istreambuf_iterator<char>(),
		'\n'
	);
	PVRush::PVUtils::sort_file(reference_file.c_str(), reference_sorted_file.c_str());

	// Count line in extracted file
	std::ifstream output_file_stream(output_file);
	if(not output_file_stream.good()) {
		return 1;
	}
	size_t output_file_line_count = std::count(
		std::istreambuf_iterator<char>(output_file_stream),
		std::istreambuf_iterator<char>(),
		'\n'
	);
	PVRush::PVUtils::sort_file(output_file.c_str());

	// Check number of line is the same with : reference_file / exported_file / count call
	PV_ASSERT_VALID((output_file_line_count == reference_file_line_count && output_file_line_count == count));

	// Checksum of reference and exported files are sames
	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(output_file.c_str(), reference_sorted_file.c_str()));

	/*
	 * Do some clean up
	 */
	std::remove(output_file.c_str());
	std::remove(reference_sorted_file.c_str());

	return 0;
}

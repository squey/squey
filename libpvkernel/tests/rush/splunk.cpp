/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#include "../../plugins/common/splunk/PVSplunkAPI.h"
#include "../../plugins/common/splunk/PVSplunkInfos.h"
#include "../../plugins/common/splunk/PVSplunkQuery.h"

#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/core/inendi_assert.h>

#include "common.h"

#include <fstream>

int main(int argc, char** argv)
{
	if (argc <= 1) {
		std::cerr << "Usage: " << argv[0] << " <query_export>" << std::endl;
		return 1;
	}

	pvtest::init_ctxt();

	/*
	 * Set up a SplunkInfo to connect with the server
	 */
	PVRush::PVSplunkInfos infos;
	infos.set_host("https://connectors.srv.picviz");
	infos.set_port(8089);
	infos.set_login("admin");
	infos.set_password("changeme");
	infos.set_splunk_index("main");
	infos.set_splunk_sourcetype("proxy_sample");
	infos.set_splunk_host("connectors2");

	/*
	 * Set up a SplunkQuery to define data to extract
	 */
	PVRush::PVSplunkQuery query(infos, "total_bytes > 5000", "");

	/*
	 * Set up the API for communication with the server
	 */
	PVRush::PVSplunkAPI splunk(infos);

	std::string error;

	/*************************************************************************
	 * Check connection is correctly done with the splunk server
	 * TODO : Check if the connection fail
	 *************************************************************************/
	bool connection_ok = splunk.check_connection(&error);
	if (not error.empty()) {
		std::cout << error << std::endl;
	}
	PV_ASSERT_VALID(connection_ok);

	/*************************************************************************
	 * Check we can get the correct indexes from the server
	 * TODO : What if there are no indexes?
	 *************************************************************************/
	PVRush::PVSplunkAPI::strings_t indexes = splunk.indexes(&error);
	if (not error.empty()) {
		std::cout << error << std::endl;
	}
	PV_ASSERT_VALID(indexes.size() == 3 && indexes[0] == "history" && indexes[1] == "main" &&
	                indexes[2] == "summary");

	/*************************************************************************
	 * Check we can get the correct sourcetypes from the server
	 * TODO : What if there are no sourcetypes?
	 *************************************************************************/
	PVRush::PVSplunkAPI::strings_t sourcetypes = splunk.sourcetypes(&error);
	if (not error.empty()) {
		std::cout << error << std::endl;
	}
	PV_ASSERT_VALID(sourcetypes.size() == 1 && sourcetypes[0] == "proxy_sample");

	/*************************************************************************
	 * Check we can get the correct hosts from the server
	 * TODO : What if there are no hosts?
	 *************************************************************************/
	PVRush::PVSplunkAPI::strings_t hosts = splunk.hosts(&error);
	if (not error.empty()) {
		std::cout << error << std::endl;
	}
	PV_ASSERT_VALID(hosts.size() == 2 && hosts[0] == "connectors" && hosts[1] == "connectors2");

	/*************************************************************************
	 * Check we can get the correct columns from the server
	 * TODO : What if there are no columns?
	 *************************************************************************/
	PVRush::PVSplunkAPI::columns_t columns = splunk.columns(&error);
	if (not error.empty()) {
		std::cout << error << std::endl;
	}

	PV_ASSERT_VALID(columns == PVRush::PVSplunkAPI::columns_t({{"category", "string"},
	                                                           {"date_hour", "integer"},
	                                                           {"date_mday", "integer"},
	                                                           {"date_minute", "integer"},
	                                                           {"date_month", "string"},
	                                                           {"date_second", "integer"},
	                                                           {"date_wday", "string"},
	                                                           {"date_year", "integer"},
	                                                           {"date_zone", "integer"},
	                                                           {"eventtype", "string"},
	                                                           {"host", "string"},
	                                                           {"http_method", "string"},
	                                                           {"http_status", "integer"},
	                                                           {"index", "string"},
	                                                           {"linecount", "integer"},
	                                                           {"login", "integer"},
	                                                           {"mime_type", "string"},
	                                                           {"punct", "string"},
	                                                           {"result_code", "string"},
	                                                           {"source", "string"},
	                                                           {"sourcetype", "string"},
	                                                           {"splunk_server", "string"},
	                                                           {"splunk_server_group", "string"},
	                                                           {"src_ip", "string"},
	                                                           {"time", "integer"},
	                                                           {"time_spent", "integer"},
	                                                           {"timeendpos", "integer"},
	                                                           {"timestartpos", "integer"},
	                                                           {"total_bytes", "integer"},
	                                                           {"url", "string"},
	                                                           {"user_agent", "string"},
	                                                           {"_bkt", "string"},
	                                                           {"_cd", "string"},
	                                                           {"_eventtype_color", "string"},
	                                                           {"_indextime", "integer"},
	                                                           {"_kv", "integer"},
	                                                           {"_raw", "string"},
	                                                           {"_serial", "integer"},
	                                                           {"_si", "string"},
	                                                           {"_sourcetype", "string"},
	                                                           {"_subsecond", "string"},
	                                                           {"_time", "integer"}}));

	/*************************************************************************
	 * Check we can correctly count number of matching fields
	 * TODO : What if there are no matching field?
	 *************************************************************************/
	size_t count = splunk.count(query, &error);
	if (not error.empty()) {
		std::cout << error << std::endl;
	}
	PV_ASSERT_VALID(count == 17774 && error.empty());

	/*************************************************************************
	 * Check we can correctly export result
	 * TODO : What if there are no matching field?
	 *************************************************************************/
	std::string output_file = pvtest::get_tmp_filename();
	std::string reference_file = argv[1];
	std::string reference_sorted_file = output_file + "_sorted";
	std::ofstream output_stream(output_file, std::ios::out | std::ios::trunc);
	std::string data;
	bool query_end = false;
	do {
		query_end = !splunk.extract(query, data, &error);
		output_stream << data.c_str();
	} while (query_end == false);

	// Count number of line in the reference file
	std::ifstream reference_file_stream(reference_file);
	if (not reference_file_stream.good()) {
		return 1;
	}
	size_t reference_file_line_count =
	    std::count(std::istreambuf_iterator<char>(reference_file_stream),
	               std::istreambuf_iterator<char>(), '\n');
	PVRush::PVUtils::sort_file(reference_file.c_str(), reference_sorted_file.c_str());

	// Count number of line in the exported file
	std::ifstream output_file_stream(output_file);
	if (not output_file_stream.good()) {
		return 1;
	}
	size_t output_file_line_count = std::count(std::istreambuf_iterator<char>(output_file_stream),
	                                           std::istreambuf_iterator<char>(), '\n');
	PVRush::PVUtils::sort_file(output_file.c_str());

	// Check there are the same number of lines for reference, exported file
	// and count API call
	PV_ASSERT_VALID(output_file_line_count == reference_file_line_count &&
	                output_file_line_count == count);

	// Check sorted content is the same as Splunk doesn't return ordered result
	PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(output_file.c_str(),
	                                                         reference_sorted_file.c_str()));

	/*
	 * Do some clean up
	 */
	std::remove(output_file.c_str());
	std::remove(reference_sorted_file.c_str());

	return 0;
}

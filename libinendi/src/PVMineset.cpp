/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include <inendi/PVMineset.h>

#include <inendi/PVView.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/core/PVConfig.h>

#include <pvcop/types/factory.h>

#include <fstream>
#include <string>

#include <stdio.h>
#include <sys/stat.h>

#include <curl/curl.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

#define MINESET_HOST                                "horus1.sgi.com" // "beta.mineset.com" confuses cURL...
constexpr const char MINESET_API_UPLOAD_DATASET[] = "https://" MINESET_HOST "/develop/api/datasets/upload-and-create";
constexpr const char MINESET_API_DATASET[]        = "https://" MINESET_HOST "/develop/dataset?id=";
constexpr const char DATASET_NAME[]               = "inendi_export";

static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp)
{
    ((std::string*)userp)->append((char*)contents, size * nmemb);

    return size * nmemb;
}

/**
 * Initializes the server communication with the proper parameters
 */
static CURL* init_curl()
{
	CURL* curl = curl_easy_init();

	/**
	 * config.ini :
	 *
	 * [mineset]
	 * login=
	 * password=
	 */
	const QSettings& pvconfig = PVCore::PVConfig::get().config();
	std::string login = pvconfig.value("mineset/login").toString().toStdString();
	std::string password = pvconfig.value("mineset/password").toString().toStdString();

	curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
	curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
	curl_easy_setopt(curl, CURLOPT_USERPWD, (login + ":" + password).c_str());
	curl_easy_setopt(curl, CURLOPT_USERAGENT, (std::string("INENDI Inspector ") + INENDI_CURRENT_VERSION_STR).c_str());
	curl_easy_setopt(curl, CURLOPT_SSLVERSION, CURL_SSLVERSION_TLSv1_2);
	curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
	curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
#ifndef NDEBUG
	curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
#endif

	return curl;
}

/**
 * Upload a compressed dataset to Mineset
 */
static std::string upload_dataset(const std::string& dataset_path)
{
	CURL* curl = init_curl();

	curl_easy_setopt(curl, CURLOPT_URL, MINESET_API_UPLOAD_DATASET);

	struct curl_httppost* post = nullptr;
	struct curl_httppost* last = nullptr;

	// dataset name
	curl_formadd(&post, &last,
			  CURLFORM_COPYNAME, "name",
			  CURLFORM_COPYCONTENTS, (std::string(DATASET_NAME) + ".tar.gz").c_str(),
			  CURLFORM_END);

	// dataset content
	curl_formadd(&post, &last,
			  CURLFORM_COPYNAME, "file",
			  CURLFORM_FILE, dataset_path.c_str(),
			  CURLFORM_END);

	curl_easy_setopt(curl, CURLOPT_HTTPPOST, post);

	std::string server_result;
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &server_result);
	CURLcode res = curl_easy_perform(curl);

	curl_formfree(post);
	curl_easy_cleanup(curl);

	if(res != CURLE_OK) {
		throw Inendi::PVMineset::mineset_error(curl_easy_strerror(res));
	}

	return server_result;
}

/**
 * Extract the dataset URL from the JSON string returned by the server on a dataset upload
 */
static std::string dataset_url(const std::string& server_result)
{
	rapidjson::Document json;
	json.Parse<0>(server_result.c_str());

	if (json.HasMember("message")) {
		if (std::string(json["message"].GetString()) == "Ok") {
			if (json.HasMember("result")) {
				if (json["result"].HasMember("id")) {
					return std::string(MINESET_API_DATASET) + std::to_string(json["result"]["id"].GetUint());
				}
			}
		}
		else if (json.HasMember("result")) {
			throw Inendi::PVMineset::mineset_error(json["message"].GetString());
		}
	}

	throw Inendi::PVMineset::mineset_error("Unable to parse server result.");
}

/**
 * Returns the Mineset JSON string representation of the format associated with the view
 */
static std::string schema(const Inendi::PVView& view)
{
	rapidjson::Document json;
	json.SetObject();
	rapidjson::Document::AllocatorType& alloc = json.GetAllocator();

	rapidjson::Value val;
	val.SetString((std::string(DATASET_NAME) + ".data").c_str(), alloc);
	json.AddMember("dataFile", val, alloc);

	rapidjson::Value json_axes(rapidjson::kArrayType);

	const Inendi::PVAxesCombination& axis_combination = view.get_axes_combination();
	for(const Inendi::PVAxis& axis : axis_combination.get_axes_list()) {
		rapidjson::Value val;
		rapidjson::Value obj;
		obj.SetObject();

		val.SetString(axis.get_name().toStdString().c_str(), alloc);
		obj.AddMember("name", val, alloc);

		std::string axis_type = axis.get_type().toStdString();
		std::string column_type;

		if (axis_type == "time") {
			column_type = "date";
		}
		else if (axis_type == "string" ||
				 axis_type == "enum" ||
				 axis_type == "host" ||
				 axis_type == "ipv4") {
			column_type = "string";
		}
		else if (axis_type == "integer") {
			column_type = "int";
		}
		else if (axis_type == "float") {
			column_type = "double";
		}
		else {
			assert(false && "Unkown axis type");
		}

		val.SetString(column_type.c_str(), alloc);
		obj.AddMember("internalType", val, alloc);

		json_axes.PushBack(obj, alloc);
	}

	json.AddMember("columns", json_axes, alloc);

	rapidjson::StringBuffer strbuf;
	rapidjson::Writer<rapidjson::StringBuffer> writer(strbuf);
	json.Accept(writer);

	return strbuf.GetString();
}

std::string Inendi::PVMineset::import_dataset(Inendi::PVView& view)
{
	const Inendi::PVSelection& sel = view.get_real_output_selection();
	PVRush::PVNraw& nraw = view.get_rushnraw_parent();

	// Create temporary directory
	std::string tmp_dir_pattern(PVRush::PVNrawCacheManager::nraw_dir().toStdString() + "/mineset_export.XXXXXXXX");
	char tmp_dir[1024];
	strcpy(tmp_dir, tmp_dir_pattern.c_str());
	mkdtemp(tmp_dir);
	QDir().mkdir((std::string(tmp_dir) + "/" + DATASET_NAME).c_str());
	std::string dataset_base_path = std::string(tmp_dir) + "/" + DATASET_NAME + "/" + DATASET_NAME;

	// Export dataset JSON schema
	std::ofstream schema_file(dataset_base_path + ".schema.json");
	schema_file << schema(view) << std::flush;

	// Export dataset content
	std::ofstream data_file(dataset_base_path + ".data");
	PVCore::PVColumnIndexes column_indexes;
	std::unordered_map<size_t, pvcop::collection::formatter_sp> datetime_formatters;
	for(const Inendi::PVAxesCombination::axes_comb_id_t& a: view.get_axes_combination().get_axes_index_list()) {
		column_indexes.push_back(a.get_axis());

		/**
		 * Convert time to ISO 8601 standard
		 */
		if (view.get_axes_combination().get_original_axis(a.get_axis()).get_type() == "time") {
			auto f = nraw.collection().formatter(a.get_axis());

			pvcop::collection::formatter_sp formatter_datetime;
			if (std::string(f->name()) == "datetime") {
				formatter_datetime = std::shared_ptr<pvcop::types::formatter_interface>(
					pvcop::types::factory::create("datetime", "%Y-%m-%dT%H:%M:%SZ"));
			}
			else if (std::string(f->name()) == "datetime_us") {
				formatter_datetime = std::shared_ptr<pvcop::types::formatter_interface>(
					pvcop::types::factory::create("datetime_us", "%Y-%m-%dT%H:%M:%S.%FZ"));
			}
			else if (std::string(f->name()) == "datetime_ms") {
				formatter_datetime = std::shared_ptr<pvcop::types::formatter_interface>(
				    pvcop::types::factory::create("datetime_ms", "yyyy-MM-dd'T'HH:mm:ss.S'Z'"));
			}
			else {
				assert(false && "Unknown datetime formatter");
			}

			datetime_formatters[a.get_axis()] = f;
			nraw.collection().set_formatter(a.get_axis(), formatter_datetime);
		}
	}

	PVRow start = 0;
	PVRow step_count = 10000;
	PVRow nrows = nraw.get_row_count();

	while (true) {
		start = sel.find_next_set_bit(start, nrows);
		if (start == PVROW_INVALID_VALUE) {
			break;
		}
		step_count = std::min(step_count, nrows - start);

		view.get_rushnraw_parent().export_lines(
				data_file,
				sel,
				column_indexes,
				start,
				step_count,
				"\t" /* = default_sep_char */
			);

		start += step_count;
	}

	// Put datetime formatters back
	for (const auto& datetime_formatter_it : datetime_formatters) {
		nraw.collection().set_formatter(datetime_formatter_it.first, datetime_formatter_it.second);
	}

	// Compress dataset
	int ret_code = system((std::string("tar -c -I pigz -f ") + tmp_dir + "/" + DATASET_NAME + ".tar.gz -C " + tmp_dir + " " + DATASET_NAME).c_str());
	if (ret_code != 0) { // no pigz support ? falling back on standard gzip compression...
		pvlogger::warn() << "Mineset export : error when compressing. Is pigz installed ? Retrying with standard gzip..." << std::endl;
		if (system((std::string("tar zcf ") + tmp_dir + "/" + DATASET_NAME + ".tar.gz -C " + tmp_dir + " " + DATASET_NAME).c_str()) != 0) {
			throw Inendi::PVMineset::mineset_error("Error when compressing dataset");
		}
	}

	// Upload compressed dataset
	std::string server_result = upload_dataset(std::string(tmp_dir) + "/" + DATASET_NAME + ".tar.gz");
	std::string url = dataset_url(server_result);

	// Cleanup temporary directory
	PVCore::PVDirectory::remove_rec(tmp_dir);

	return url;
}

void Inendi::PVMineset::delete_dataset(const std::string& dataset_url)
{
	CURL* curl = init_curl();

	curl_easy_setopt(curl, CURLOPT_URL, dataset_url.c_str());
	curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");

	CURLcode res = curl_easy_perform(curl);

	if(res != CURLE_OK) {
		pvlogger::error() << curl_easy_strerror(res) << std::endl;
	}

	curl_easy_cleanup(curl);
}

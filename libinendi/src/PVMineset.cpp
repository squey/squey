/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include <inendi/PVMineset.h>
#include <inendi/PVView.h>
#include <inendi/PVSource.h>

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVNraw.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/core/PVConfig.h>
#include <pvkernel/rush/PVCSVExporter.h>

#include <pvcop/types/factory.h>
#include <pvcop/types/formatter/formatter_interface.h>

#include <fstream>
#include <string>

#include <stdio.h>
#include <sys/stat.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

constexpr const char DATASET_NAME[] = "inendi_export";

static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp)
{
	((std::string*)userp)->append((char*)contents, size * nmemb);

	return size * nmemb;
}

/**
 * Initializes the server communication with the proper parameters
 */
CURL* Inendi::PVMineset::init_curl()
{
	CURL* curl = curl_easy_init();

	curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
	curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BASIC);
	curl_easy_setopt(curl, CURLOPT_USERPWD, (_login + ":" + _password).c_str());
	curl_easy_setopt(curl, CURLOPT_USERAGENT,
	                 (std::string("INENDI Inspector ") + INENDI_CURRENT_VERSION_STR).c_str());
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
std::string Inendi::PVMineset::upload_dataset(const std::string& dataset_path)
{
	CURL* curl = init_curl();

	const std::string MINESET_API_UPLOAD_DATASET = _url + "/api/datasets/upload-and-create";
	curl_easy_setopt(curl, CURLOPT_URL, MINESET_API_UPLOAD_DATASET.c_str());

	struct curl_httppost* post = nullptr;
	struct curl_httppost* last = nullptr;

	// dataset name
	curl_formadd(&post, &last, CURLFORM_COPYNAME, "name", CURLFORM_COPYCONTENTS,
	             (std::string(DATASET_NAME) + ".tar.gz").c_str(), CURLFORM_END);

	// dataset content
	curl_formadd(&post, &last, CURLFORM_COPYNAME, "file", CURLFORM_FILE, dataset_path.c_str(),
	             CURLFORM_END);

	curl_easy_setopt(curl, CURLOPT_HTTPPOST, post);

	std::string server_result;
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &server_result);
	CURLcode res = curl_easy_perform(curl);

	curl_formfree(post);
	curl_easy_cleanup(curl);

	if (res != CURLE_OK) {
		throw Inendi::PVMineset::mineset_error(curl_easy_strerror(res));
	}

	return server_result;
}

std::string Inendi::PVMineset::dataset_url(const std::string& server_result)
{
	rapidjson::Document json;
	json.Parse<0>(server_result.c_str());

	std::string const MINESET_API_DATASET = _url + "/dataset?id=";

	if (json.HasMember("message")) {
		if (std::string(json["message"].GetString()) == "Ok") {
			if (json.HasMember("result") and json["result"].HasMember("id")) {
				return MINESET_API_DATASET + std::to_string(json["result"]["id"].GetUint());
			}
		} else if (json.HasMember("result")) {
			throw Inendi::PVMineset::mineset_error(json["message"].GetString());
		}
	}

	throw Inendi::PVMineset::mineset_error("Unable to parse server result.");
}

/**
 * Returns the Mineset JSON string representation of the format associated with
 * the view
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
	for (PVCol i : axis_combination.get_combination()) {
		const PVRush::PVAxisFormat axis = axis_combination.get_axis(i);
		rapidjson::Value val;
		rapidjson::Value obj;
		obj.SetObject();

		val.SetString(axis.get_name().toStdString().c_str(), alloc);
		obj.AddMember("name", val, alloc);

		std::string axis_type = axis.get_type().toStdString();
		std::string column_type;

		if (axis_type == "time") {
			column_type = "date";
		} else if (axis_type == "number_int32" or axis_type == "number_uint32") {
			column_type = "int";
		} else if (axis_type == "number_float" or axis_type == "number_double") {
			column_type = "double";
		} else {
			column_type = "string"; // fallback on string type for other types
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

/**
 * Compress data in tmp_dir and return zip file path.
 */
static std::string compress_data(std::string const& tmp_dir)
{
	// Compress dataset
	std::string dataset_zip_path = tmp_dir + "/" + DATASET_NAME + ".tar.gz";

	int ret_code = system((std::string("tar -c -I pigz -f ") + dataset_zip_path + " -C " + tmp_dir +
	                       " " + DATASET_NAME)
	                          .c_str());
	if (ret_code != 0) { // no pigz support ? falling back on standard gzip compression...
		pvlogger::warn() << "Mineset export : error when compressing. Is pigz "
		                    "installed ? Retrying with standard gzip..."
		                 << std::endl;
		if (system(
		        (std::string("tar zcf ") + dataset_zip_path + " -C " + tmp_dir + " " + DATASET_NAME)
		            .c_str()) != 0) {
			throw Inendi::PVMineset::mineset_error("Error when compressing dataset");
		}
	}
	return dataset_zip_path;
}

// Use anonymous namespace to keep this class local.
namespace
{
/**
 * RAII class that change NRaw format for times to match mineset format and set
 * back its correct values at the end.
 */
class LocalMinesetFormat
{
  public:
	/**
	 * Update format to match Mineset datetime format.
	 */
	LocalMinesetFormat(Inendi::PVView& view) : _view(view)
	{
		PVRush::PVNraw& nraw = view.get_rushnraw_parent();

		for (PVRush::PVAxisFormat const& axis :
		     view.get_parent<Inendi::PVSource>().get_format().get_axes()) {
			/**
			 * Convert time to ISO 8601 standard
			 */
			if (axis.get_type() == "time") {
				auto f = nraw.column(axis.index).formatter();

				pvcop::types::formatter_interface::shared_ptr formatter_datetime;
				if (std::string(f->name()) == "datetime") {
					formatter_datetime = std::shared_ptr<pvcop::types::formatter_interface>(
					    pvcop::types::factory::create("datetime", "%Y-%m-%dT%H:%M:%SZ"));
				} else if (std::string(f->name()) == "datetime_us") {
					formatter_datetime = std::shared_ptr<pvcop::types::formatter_interface>(
					    pvcop::types::factory::create("datetime_us", "%Y-%m-%dT%H:%M:%S.%FZ"));
				} else {
					assert(std::string(f->name()) == "datetime_ms" && "Unknown datetime formatter");
					formatter_datetime = std::shared_ptr<pvcop::types::formatter_interface>(
					    pvcop::types::factory::create("datetime_ms", "yyyy-MM-dd'T'HH:mm:ss.S'Z'"));
				}

				_datetime_formatters[axis.index] = f;
				const_cast<pvcop::db::array&>(nraw.column(axis.index))
				    .set_formatter(formatter_datetime);
			}
		}
	}

	/**
	 * Set back NRaw format for time.
	 */
	~LocalMinesetFormat()
	{
		PVRush::PVNraw& nraw = _view.get_rushnraw_parent();

		// Put datetime formatters back
		for (const auto& datetime_formatter_it : _datetime_formatters) {
			const_cast<pvcop::db::array&>(nraw.column(PVCol(datetime_formatter_it.first)))
			    .set_formatter(datetime_formatter_it.second);
		}
	}

  private:
	std::unordered_map<size_t, pvcop::types::formatter_interface::shared_ptr>
	    _datetime_formatters; //!< Updates format.
	Inendi::PVView& _view;    //!< Changed view.
};
}

/**
 * config.ini :
 *
 * [mineset]
 * login=
 * password=
 * url =
 */
Inendi::PVMineset::PVMineset()
    : _login(PVCore::PVConfig::get().config().value("mineset/login").toString().toStdString())
    , _password(PVCore::PVConfig::get().config().value("mineset/password").toString().toStdString())
    , _url(PVCore::PVConfig::get().config().value("mineset/url").toString().toStdString())
{
}

bool Inendi::PVMineset::is_enabled()
{
	return PVCore::PVConfig::get().config().childGroups().contains("mineset");
}

/**
 * Import dataset from inspector to Mineset.
 */
std::string Inendi::PVMineset::import_dataset(Inendi::PVView& view)
{
	const Inendi::PVSelection& sel = view.get_real_output_selection();
	PVRush::PVNraw& nraw = view.get_rushnraw_parent();

	// Create temporary directory
	std::string tmp_dir_pattern(PVRush::PVNrawCacheManager::nraw_dir().toStdString() +
	                            "/mineset_export.XXXXXXXX");
	char tmp_dir[1024];
	strcpy(tmp_dir, tmp_dir_pattern.c_str());
	mkdtemp(tmp_dir);
	QDir().mkdir((std::string(tmp_dir) + "/" + DATASET_NAME).c_str());
	std::string dataset_base_path = std::string(tmp_dir) + "/" + DATASET_NAME + "/" + DATASET_NAME;

	// Export dataset JSON schema
	std::ofstream schema_file(dataset_base_path + ".schema.json");
	schema_file << schema(view) << std::flush;

	// Export dataset content
	{

		LocalMinesetFormat lf(view);

		PVCore::PVColumnIndexes column_indexes = view.get_axes_combination().get_combination();

		PVRush::PVCSVExporter::export_func_f export_func =
		    [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
		        const std::string& quote) { return nraw.export_line(row, cols, sep, quote); };
		PVRush::PVCSVExporter exp(column_indexes, nraw.row_count(), export_func,
		                          "\t" /* = default_sep_char */);
		exp.export_rows(dataset_base_path + ".data", sel);
	}

	// Compress dataset
	std::string dataset_zip_path = compress_data(tmp_dir);

	// Upload compressed dataset
	std::string server_result = instance().upload_dataset(dataset_zip_path);
	std::string url = instance().dataset_url(server_result);

	// Cleanup temporary directory
	PVCore::PVDirectory::remove_rec(tmp_dir);

	return url;
}

void Inendi::PVMineset::delete_dataset(const std::string& dataset_url)
{
	CURL* curl = instance().init_curl();

	curl_easy_setopt(curl, CURLOPT_URL, dataset_url.c_str());
	curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");

	CURLcode res = curl_easy_perform(curl);

	if (res != CURLE_OK) {
		pvlogger::error() << curl_easy_strerror(res) << std::endl;
	}

	curl_easy_cleanup(curl);
}

Inendi::PVMineset& Inendi::PVMineset::instance()
{
	static Inendi::PVMineset sing;
	return sing;
}

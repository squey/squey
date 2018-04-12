/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */

#ifndef __PVKERNEL_PVLICENSEACTIVATOR_H__
#define __PVKERNEL_PVLICENSEACTIVATOR_H__

#include <pvkernel/core/PVUtils.h>

#include <curl/curl.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>

#include <QDir>
#include <QFileInfo>
#include <QString>

#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <unordered_map>

// REST API documentation : http://34.245.246.226:8080/licensing/docs/

namespace PVCore
{

#define INSPECTOR_REST_API_DOMAIN "http://34.245.246.226"
#ifdef INENDI_DEVELOPER_MODE
#define INSPECTOR_REST_API_PORT ":8080"
#define INSPECTOR_REST_API_AUTH_TOKEN "1ae64b54eddfb65d65dc7da6d6891624f039e256"
#else // USER_TARGET=customer
#define INSPECTOR_REST_API_PORT ""
#define INSPECTOR_REST_API_AUTH_TOKEN "b170327aa762b93dedd506b5fc9d9539f899ee55"
#endif
#define INSPECTOR_REST_API_SOCKET INSPECTOR_REST_API_DOMAIN INSPECTOR_REST_API_PORT

static size_t curl_write_callback(void* contents, size_t size, size_t nmemb, void* userp)
{
	((std::string*)userp)->append((char*)contents, size * nmemb);

	return size * nmemb;
}

class PVLicenseActivator
{
  public:
	enum class EError {
		// Both license types
		NO_ERROR,
		NO_INTERNET_CONNECTION,
		ACTIVATION_SERVICE_UNAVAILABLE,
		UNABLE_TO_READ_LICENSE_FILE,
		UNABLE_TO_INSTALL_LICENSE_FILE,

		// Trial license type only
		UNKOWN_USER,
		USER_NOT_VALIDATED,
		TRIAL_ALREADY_ACTIVATED,

		// Paid license type only
		UNKNOWN_ACTIVATION_KEY,
		ACTIVATION_KEY_ALREADY_ACTIVATED,

	};

  public:
	PVLicenseActivator(const std::string& inendi_license_path)
	    : _inendi_license_path(inendi_license_path)
	{
	}

  public:
	EError online_activation(const std::string& email,
	                         const std::string& locking_code,
	                         const std::string& activation_key) const
	{
		std::unique_ptr<CURL, std::function<void(CURL*)>> curl(
		    curl_easy_init(), [](CURL* curl) { curl_easy_cleanup(curl); });

		curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
		curl_easy_setopt(curl.get(), CURLOPT_SSLVERSION, CURL_SSLVERSION_TLSv1_2);
#ifndef NDEBUG
		curl_easy_setopt(curl.get(), CURLOPT_VERBOSE, 1L);
#endif

		const std::string INSPECTOR_REST_API_ENDPOINT =
		    std::string(INSPECTOR_REST_API_SOCKET) + "/licensing/activate_license/";
		curl_easy_setopt(curl.get(), CURLOPT_URL, INSPECTOR_REST_API_ENDPOINT.c_str());

		std::unique_ptr<curl_slist, std::function<void(curl_slist*)>> headers(
		    nullptr, [](curl_slist* headers) { curl_slist_free_all(headers); });
		std::string auth_token_header{std::string("Authorization: Token ") +
		                              INSPECTOR_REST_API_AUTH_TOKEN};
		std::string content_type_header{"Content-Type: application/json"};
		curl_slist* headers_list = nullptr;
		headers_list = curl_slist_append(headers_list, auth_token_header.c_str());
		headers_list = curl_slist_append(headers_list, content_type_header.c_str());
		headers.reset(headers_list);

		curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers.get());

		std::string post_body{
		    std::string("{") +
		    (email.empty() ? "" : (std::string("   \"email\":\"") + email + "\",")) +
		    "   \"locking_code\":\"" + locking_code + "\"," + "   \"activation_key\":\"" +
		    activation_key + "\""
		                     "}"};

		curl_easy_setopt(curl.get(), CURLOPT_HTTPPOST, true);
		curl_easy_setopt(curl.get(), CURLOPT_COPYPOSTFIELDS, post_body.c_str());

		std::string result_content;
		curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &result_content);
		curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, &curl_write_callback);

		CURLcode curl_result_code = curl_easy_perform(curl.get());

		if (curl_result_code != CURLE_OK) {
			return EError::NO_INTERNET_CONNECTION;
		}

		rapidjson::Document json;
		json.Parse<0>(result_content.c_str());
		if (json.IsObject() and json.HasMember("error")) {
			return _server_errors.at(json["error"].GetString());
		}

		assert(json.IsObject() and json.HasMember("license"));

		// Install license content
		if (not ensure_license_folder_exists()) {
			return EError::UNABLE_TO_INSTALL_LICENSE_FILE;
		}
		std::ofstream output_stream(_inendi_license_path);
		output_stream << json["license"].GetString();
		if (output_stream.bad()) {
			return EError::UNABLE_TO_INSTALL_LICENSE_FILE;
		}

		return EError::NO_ERROR;
	}

  public:
	EError offline_activation(const QString& user_license_path) const
	{
		assert(not user_license_path.isEmpty());

		if (not QFileInfo(user_license_path).isReadable()) {
			return EError::UNABLE_TO_READ_LICENSE_FILE;
		}

		if (not ensure_license_folder_exists() or
		    not QFile::copy(user_license_path, QString::fromStdString(_inendi_license_path))) {
			return EError::UNABLE_TO_INSTALL_LICENSE_FILE;
		}

		return EError::NO_ERROR;
	}

  public:
	static std::string get_locking_code()
	{
		const std::string& echoid_output = PVCore::exec_cmd("inendi-echoid");
		std::string locking_code;
		std::istringstream iss(echoid_output);
		std::string line;

		while (std::getline(iss, line)) {
			if (line.find("Locking Code 1     : ") != std::string::npos) {
				return line.substr(23, 24);
			}
		}

		return {};
	}

  private:
	bool ensure_license_folder_exists() const
	{
		QString license_folder =
		    QFileInfo(QString::fromStdString(_inendi_license_path)).dir().path();
		if (not QFileInfo(license_folder).exists()) {
			return QDir().mkpath(license_folder);
		}
		return true;
	}

  private:
	const std::unordered_map<std::string, EError> _server_errors{
	    {"ACTIVATION_SERVICE_UNAVAILABLE", EError::ACTIVATION_SERVICE_UNAVAILABLE},
	    {"UNKOWN_USER", EError::UNKOWN_USER},
	    {"USER_NOT_VALIDATED", EError::USER_NOT_VALIDATED},
	    {"TRIAL_ALREADY_ACTIVATED", EError::TRIAL_ALREADY_ACTIVATED},
	    {"UNKNOWN_ACTIVATION_KEY", EError::UNKNOWN_ACTIVATION_KEY},
	    {"ACTIVATION_KEY_ALREADY_ACTIVATED", EError::ACTIVATION_KEY_ALREADY_ACTIVATED},
	};

  private:
	std::string _inendi_license_path;
};
}

#endif // __PVKERNEL_PVLICENSEACTIVATOR_H__

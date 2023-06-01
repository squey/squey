/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __PVCRASHREPORTSENDER_H__
#define __PVCRASHREPORTSENDER_H__

#include "PVCrashReporterToken.h"

#include <memory>
#include <string>

#include <curl/curl.h>

#include <rapidjson/document.h>

#include <QFileInfo>

#include <sys/stat.h>

#include <pvlogger.h>

#define SEND_CRASH_REPORTS_AS_GITLAB_ISSUES 0

static size_t write_callback(void* contents, size_t size, size_t nmemb, void* userp)
{
	((std::string*)userp)->append((char*)contents, size * nmemb);

	return size * nmemb;
}

namespace PVCore
{

class PVCrashReportSender
{
  public:
	static int send(const std::string& minidump_path,
	                const std::string& version)
	{
		struct stat file_info;
		FILE* fd = fopen(minidump_path.c_str(), "rb");
		fstat(fileno(fd), &file_info);

		std::unique_ptr<CURL, std::function<void(CURL*)>> curl(
		    curl_easy_init(), [](CURL* curl) { curl_easy_cleanup(curl); });

		curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, 60L);
		curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
		curl_easy_setopt(curl.get(), CURLOPT_SSLVERSION, CURL_SSLVERSION_MAX_DEFAULT);
#ifndef NDEBUG
		curl_easy_setopt(curl.get(), CURLOPT_VERBOSE, 1L);
#endif

#if SEND_CRASH_REPORTS_AS_GITLAB_ISSUES
		constexpr static std::string_view SQUEY_GITLAB_API_ENDPOINT = "https://gitlab.com/api/v4/projects/squey";
		constexpr static std::string_view SQUEY_GITLAB_API_ISSUES_TITLE = "Crash report";

		std::unique_ptr<curl_slist, std::function<void(curl_slist*)>> headers(
		    nullptr, [](curl_slist* headers) { curl_slist_free_all(headers); });
		std::string auth_token_header{std::string("PRIVATE-TOKEN: ").append(SQUEY_CRASH_REPORTER_TOKEN)};
		curl_slist* headers_list = nullptr;
		headers_list = curl_slist_append(headers_list, auth_token_header.c_str());
		headers.reset(headers_list);

		curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers.get());
		curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, write_callback);
		std::string result;
		curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &result);

		std::unique_ptr<curl_httppost, std::function<void(curl_httppost*)>> postvars(
			nullptr, [](curl_httppost* formpost) { curl_formfree(formpost); });

		// Upload crash report
		const std::string SQUEY_GITLAB_API_UPLOAD_ENDPOINT =
		    std::string(SQUEY_GITLAB_API_ENDPOINT) + "/uploads";
		curl_easy_setopt(curl.get(), CURLOPT_URL, SQUEY_GITLAB_API_UPLOAD_ENDPOINT.c_str());

		struct curl_httppost* formpost = nullptr;
		struct curl_httppost* lastptr = nullptr;

		curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, "file", CURLFORM_FILE,
				minidump_path.c_str(), CURLFORM_END);

		postvars.reset(formpost);
		curl_easy_setopt(curl.get(), CURLOPT_HTTPPOST, postvars.get());

		/*CURLcode curl_ret =*/ curl_easy_perform(curl.get());

		long http_code = 0;
		curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_code);

		if (http_code != 201) {
			return http_code;
		}

		rapidjson::Document json;
		json.Parse<0>(result.c_str());
		std::string minidump_link(json["markdown"].GetString());

		// Create Issue
		const std::string SQUEY_GITLAB_API_ISSUES_ENDPOINT =
		    std::string(SQUEY_GITLAB_API_ENDPOINT) + "/issues/";
		curl_easy_setopt(curl.get(), CURLOPT_URL, SQUEY_GITLAB_API_ISSUES_ENDPOINT.c_str());

		std::string description = minidump_link + " (version " + version + ")";

		formpost = nullptr;
		lastptr = nullptr;

		curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, "title", CURLFORM_COPYCONTENTS,
		             SQUEY_GITLAB_API_ISSUES_TITLE.data(), CURLFORM_END);
		curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, "issue_type", CURLFORM_COPYCONTENTS,
		             "incident", CURLFORM_END);
		curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, "confidential", CURLFORM_COPYCONTENTS,
					 "true", CURLFORM_END);
		curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, "labels", CURLFORM_COPYCONTENTS,
					 "kind::crash", CURLFORM_END);
		curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, "description", CURLFORM_COPYCONTENTS,
					 description.c_str(), CURLFORM_END);
					 
		postvars.reset(formpost);
		curl_easy_setopt(curl.get(), CURLOPT_HTTPPOST, postvars.get());

		/*CURLcode curl_ret =*/ curl_easy_perform(curl.get());

		http_code = 0;
		curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_code);

		if (http_code == 201) {
			return 0;
		} else {
			return http_code;
		}
#else // Send crash reports to bugsplat.com
		#define SQUEY_BUGSPLAT_DATABASE "squey"
		#define SQUEY_BUGSPLAT_API_ENDPOINT "https://" SQUEY_BUGSPLAT_DATABASE ".bugsplat.com/post/bp/crash/crashpad.php"

		// Compress minidump file to speed upload
		std::string compressed_minidump_path(minidump_path + ".zip");
		std::string zip_cmd = std::string("zip ") + compressed_minidump_path + " " + minidump_path;
		system(zip_cmd.c_str());

		curl_easy_setopt(curl.get(), CURLOPT_ACCEPT_ENCODING, "zstd, br, gzip, deflate");
		curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, write_callback);
		std::string result;
		curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &result);

		std::unique_ptr<curl_httppost, std::function<void(curl_httppost*)>> postvars(
			nullptr, [](curl_httppost* formpost) { curl_formfree(formpost); });

		struct curl_httppost* formpost = nullptr;
		struct curl_httppost* lastptr = nullptr;

		curl_easy_setopt(curl.get(), CURLOPT_URL, SQUEY_BUGSPLAT_API_ENDPOINT);
		curl_formadd(
			&formpost,
			&lastptr,
			CURLFORM_COPYNAME,
			"upload_file_minidump",
			CURLFORM_FILE,
			compressed_minidump_path.c_str(),
			CURLFORM_END
		);
		curl_formadd(
			&formpost,
			&lastptr,
			CURLFORM_COPYNAME,
			"product",
			CURLFORM_COPYCONTENTS,
			"Squey",
			CURLFORM_END
		);
		curl_formadd(
			&formpost,
			&lastptr,
			CURLFORM_COPYNAME,
			"version",
			CURLFORM_COPYCONTENTS,
			version.c_str(),
			CURLFORM_END
		);

		postvars.reset(formpost);
		curl_easy_setopt(curl.get(), CURLOPT_HTTPPOST, postvars.get());

		// Upload crash report
		curl_easy_perform(curl.get());

		long http_code = 0;
		curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_code);

		if (http_code == 200) {
			return 0;
		} else {
			return http_code;
		}
#endif
	}

	static bool test_auth()
	{
#if SEND_CRASH_REPORTS_AS_GITLAB_ISSUES
		std::unique_ptr<CURL, std::function<void(CURL*)>> curl(
		    curl_easy_init(), [](CURL* curl) { curl_easy_cleanup(curl); });

		curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, 10L);
		curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
		curl_easy_setopt(curl.get(), CURLOPT_SSLVERSION, CURL_SSLVERSION_TLSv1_2);
#ifndef NDEBUG
		curl_easy_setopt(curl.get(), CURLOPT_VERBOSE, 1L);
#endif

		std::unique_ptr<curl_slist, std::function<void(curl_slist*)>> headers(
		    nullptr, [](curl_slist* headers) { curl_slist_free_all(headers); });
		std::string auth_token_header{std::string("PRIVATE-TOKEN: ").append(SQUEY_CRASH_REPORTER_TOKEN)};
		curl_slist* headers_list = nullptr;
		headers_list = curl_slist_append(headers_list, auth_token_header.c_str());
		headers.reset(headers_list);

		curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers.get());
		curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, write_callback);
		std::string result;
		curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &result);

		curl_easy_setopt(curl.get(), CURLOPT_URL, "https://gitlab.com/api/v4/personal_access_tokens");

		/*CURLcode curl_ret =*/ curl_easy_perform(curl.get());

		long http_code = 0;
		curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_code);

		return http_code == 200;
#else
		return true;
#endif
	}
};
} // namespace PVCore

#endif // __PVCRASHREPORTSENDER_H__

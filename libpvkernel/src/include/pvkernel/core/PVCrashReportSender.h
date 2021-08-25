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

#include <memory>
#include <string>

#include <curl/curl.h>

#include <pvkernel/core/PVRESTAPI.h>

#include <QFileInfo>

#include <sys/stat.h>

namespace PVCore
{

class PVCrashReportSender
{
  public:
	static int send(const std::string& minidump_path,
	                const std::string& version,
	                const std::string& locking_code)
	{
		struct stat file_info;
		FILE* fd = fopen(minidump_path.c_str(), "rb");
		fstat(fileno(fd), &file_info);

		std::unique_ptr<CURL, std::function<void(CURL*)>> curl(
		    curl_easy_init(), [](CURL* curl) { curl_easy_cleanup(curl); });

		curl_easy_setopt(curl.get(), CURLOPT_TIMEOUT, 10L);
		curl_easy_setopt(curl.get(), CURLOPT_FOLLOWLOCATION, 1L);
		curl_easy_setopt(curl.get(), CURLOPT_SSLVERSION, CURL_SSLVERSION_TLSv1_2);
#ifndef NDEBUG
		curl_easy_setopt(curl.get(), CURLOPT_VERBOSE, 1L);
#endif

		const std::string INSPECTOR_REST_API_ENDPOINT =
		    std::string(INSPECTOR_REST_API_SOCKET) + "/report/crash/";
		curl_easy_setopt(curl.get(), CURLOPT_URL, INSPECTOR_REST_API_ENDPOINT.c_str());

		std::unique_ptr<curl_slist, std::function<void(curl_slist*)>> headers(
		    nullptr, [](curl_slist* headers) { curl_slist_free_all(headers); });
		std::string auth_token_header{std::string("Authorization: Token ") +
		                              INSPECTOR_REST_API_AUTH_TOKEN};
		curl_slist* headers_list = nullptr;
		headers_list = curl_slist_append(headers_list, auth_token_header.c_str());
		headers.reset(headers_list);

		curl_easy_setopt(curl.get(), CURLOPT_HTTPHEADER, headers.get());

		std::unique_ptr<curl_httppost, std::function<void(curl_httppost*)>> postvars(
		    nullptr, [](curl_httppost* formpost) { curl_formfree(formpost); });

		struct curl_httppost* formpost = nullptr;
		struct curl_httppost* lastptr = nullptr;

		curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, "minidump", CURLFORM_FILE,
		             minidump_path.c_str(), CURLFORM_END);

		curl_formadd(
		    &formpost, &lastptr, CURLFORM_COPYNAME, "minidump_name", CURLFORM_COPYCONTENTS,
		    QFileInfo(QString::fromStdString(minidump_path)).fileName().toStdString().c_str(),
		    CURLFORM_END);

		curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, "version", CURLFORM_COPYCONTENTS,
		             version.c_str(), CURLFORM_END);

		curl_formadd(&formpost, &lastptr, CURLFORM_COPYNAME, "locking_code", CURLFORM_COPYCONTENTS,
		             locking_code.c_str(), CURLFORM_END);

		postvars.reset(formpost);
		curl_easy_setopt(curl.get(), CURLOPT_HTTPPOST, postvars.get());

		/*CURLcode curl_ret =*/ curl_easy_perform(curl.get());

		long http_code = 0;
		curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_code);

		if (http_code == 200) {
			return 0;
		} else {
			return http_code;
		}
	}
};
} // namespace PVCore

#endif // __PVCRASHREPORTSENDER_H__

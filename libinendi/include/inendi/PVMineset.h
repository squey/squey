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

#ifndef __INENDI_PVMINESET_H__
#define __INENDI_PVMINESET_H__

#include <curl/curl.h>

#include <stdexcept>

namespace Inendi
{

class PVView;

/**
 * This class is intended to control a Mineset instance through its REST API
 */
class PVMineset
{

  public:
	/**
	 * This exception is raised when an error during the
	 * communication with Mineset occurs
	 *
	 * its what() function returns the error message returned
	 * by the server
	 */
	struct mineset_error : public std::runtime_error {
		using std::runtime_error::runtime_error;
	};

	/**
	 * Returns true if the config contains a "[mineset]" group
	 */
	static bool is_enabled();

	/**
	 * Extracts the selected rows of a view with its
	 * axes combination and import it to Mineset as a dataset
	 *
	 * @return the URL of the dataset
	 */
	static std::string import_dataset(Inendi::PVView& view);

	/**
	 * Deletes a dataset by specifying its URL
	 *
	 * @todo : This one is not called for now as PVView is never destroyed.
	 * When it will be done, we should ask if we really want to destroy dataset
	 * as we may keep these for future investigation reload.
	 */
	static void delete_dataset(const std::string& dataset_url);

  private:
	/**
	 * Set connection information for config file.
	 */
	PVMineset();

	/**
	 * Singleton instance of the server connection.
	 */
	static PVMineset& instance();

	/**
	 * Start communication with the server.
	 */
	CURL* init_curl();

	/**
	 * Upload a compressed dataset to Mineset.
	 */
	std::string upload_dataset(const std::string& dataset_path);

	/**
	 * Extract the dataset URL from the JSON string returned by the server on a dataset upload
	 */
	std::string dataset_url(const std::string& server_result);

  private:
	std::string _login;    //!< Connection login.
	std::string _password; //!< Connection password.
	std::string _url;      //!< Mineset instance API url
};
}

#endif // __INENDI_PVMINESET_H__

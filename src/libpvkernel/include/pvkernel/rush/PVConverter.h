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

#ifndef PVRUSHCONVERTER_H
#define PVRUSHCONVERTER_H

#include <cassert>
#include <exception>

#include <unicode/ucnv.h>

using namespace icu_75;

namespace PVRush
{

/**
 * Exception on invalide ICU converter creation.
 */
class PVConverterCreationError : public std::runtime_error
{
	using std::runtime_error::runtime_error;
};

/**
 * Wrapper for ICU converter to improve memory handling.
 */
class PVConverter
{

  public:
	/**
	 * Create an ICU converter from a string
	 */
	PVConverter(std::string const& converter_name)
	{
		UErrorCode status = U_ZERO_ERROR;
		_ucnv = ucnv_open(converter_name.c_str(), &status);

		if (U_FAILURE(status)) {
			// Force "UTF-8" charset if uchardet encoding is not supported by ICU
			status = U_ZERO_ERROR;
			_ucnv = ucnv_open("UTF-8", &status);

			// Throw an exception if it still fails
			if (U_FAILURE(status)) {
				throw PVConverterCreationError("Unsupported charset encoding '" + converter_name +
				                               "'");
			}
		}
	}

	/**
	 * Wrap a converter RAII style to properly handle deallocation
	 */
	PVConverter(UConverter* converter) : _ucnv(converter) {}

	/**
	 * Realease ICU converter.
	 */
	~PVConverter() { ucnv_close(_ucnv); }

	/**
	 * Direct access to ICU converter.
	 */
	UConverter& get()
	{
		assert(_ucnv);
		return *_ucnv;
	}

  private:
	UConverter* _ucnv; //!< The real ICU converter.
};
} // namespace PVRush

#endif

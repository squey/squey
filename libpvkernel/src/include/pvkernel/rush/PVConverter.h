/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVRUSHCONVERTER_H
#define PVRUSHCONVERTER_H

#include <cassert>
#include <exception>

extern "C" {
#include <unicode/ucnv.h>
}

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
	 * Create an ICU converter
	 */
	PVConverter(std::string const& converter_name)
	{
		UErrorCode status = U_ZERO_ERROR;
		_ucnv = ucnv_open(converter_name.c_str(), &status);

		if (U_FAILURE(status)) {
			throw PVConverterCreationError("Fail to create ICU converter.");
		}
	}

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
}

#endif

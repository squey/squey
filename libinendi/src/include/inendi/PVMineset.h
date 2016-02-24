/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#ifndef __INENDI_PVMINETSET_H__
#define __INENDI_PVMINETSET_H__

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
	struct mineset_error : public std::runtime_error
	{
		using std::runtime_error::runtime_error;
	};

public:
	/**
	 * Extracts the selected rows of a view with its
	 * axes combination and import it to Mineset as a dataset
	 *
	 * @return the URL of the dataset
	 */
	static std::string import_dataset(Inendi::PVView& view);

	/**
	 * Deletes a dataset by specifying its URL
	 */
	static void delete_dataset(const std::string& dataset_url);
};

}

#endif // __INENDI_PVMINETSET_H__

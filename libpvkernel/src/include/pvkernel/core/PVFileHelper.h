/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVFILEHELPER_H
#define PVCORE_PVFILEHELPER_H

namespace PVCore
{

struct PVFileHelper {

	static bool is_already_opened(const char* file_name);
};
}

#endif // PVCORE_PVFILEHELPER_H

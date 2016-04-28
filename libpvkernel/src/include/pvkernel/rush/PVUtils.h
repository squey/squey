/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_PVUTILS_H
#define PVRUSH_PVUTILS_H

class QString;
class QStringList;
#include <QByteArray>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVAxesIndexType.h>
#include <pvkernel/rush/PVNraw.h>

#include <string.h>

namespace PVRush
{
namespace PVUtils
{
bool files_have_same_content(const std::string& path1, const std::string& path2);

/**
 * Alphabetically sort the lines of a text file
 *
 * @param input_file input file path
 * @param output_file output file path (inplace sort if not specified)
 */
void sort_file(const char* input_file, const char* output_file = nullptr);

std::string
safe_export(std::string str, const std::string& sep_char, const std::string& quote_char);
void safe_export(QStringList& str_list, const std::string& sep_char, const std::string& quote_char);
}
}

#endif /* PVRUSH_PVUTILS_H */

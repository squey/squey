/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef __PVCORE_PVEXPORTER_H__
#define __PVCORE_PVEXPORTER_H__

#include <pvkernel/core/PVColumnIndexes.h>

#include <fstream>
#include <functional>
#include <string>

namespace PVCore
{
class PVSelBitField;

class PVExporter
{
  public:
	static const std::string default_sep_char;
	static const std::string default_quote_char;

  public:
	using export_func = std::function<std::string(
	    PVRow, const PVCore::PVColumnIndexes&, const std::string&, const std::string&)>;

  public:
	PVExporter(std::ostream& os,
	           const PVCore::PVSelBitField& sel,
	           const PVCore::PVColumnIndexes& column_indexes,
	           PVRow step_count,
	           export_func f,
	           const std::string& sep_char = default_sep_char,
	           const std::string& quote_char = default_quote_char);

	void export_rows(size_t start_index);

  private:
	std::ostream& _os;
	const PVCore::PVSelBitField& _sel;
	const PVCore::PVColumnIndexes& _column_indexes;
	PVRow _step_count;
	const std::string _sep_char;
	const std::string _quote_char;
	export_func _f;
};

} // namespace PVCore

#endif // __PVCORE_PVEXPORTER_H__

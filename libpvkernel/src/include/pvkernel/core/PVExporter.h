/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef __PVCORE_PVEXPORTER_H__
#define __PVCORE_PVEXPORTER_H__

#include <pvkernel/core/PVColumnIndexes.h> // for PVColumnIndexes
#include <pvkernel/core/PVStreamingCompressor.h>

#include "pvbase/types.h" // for PVRow

#include <atomic>
#include <cstddef>    // for size_t
#include <fstream>    // for ostream
#include <functional> // for function
#include <string>     // for string
#include <thread>     // for std::thread
#include <unordered_map>

namespace PVCore
{
class PVSelBitField;

struct PVExportError : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

class PVExporter
{
  public:
	static const std::string default_sep_char;
	static const std::string default_quote_char;

  public:
	using export_func = std::function<std::string(
	    PVRow, const PVCore::PVColumnIndexes&, const std::string&, const std::string&)>;

  public:
	PVExporter(const std::string& file_path,
	           const PVCore::PVSelBitField& sel,
	           const PVCore::PVColumnIndexes& column_indexes,
	           PVRow total_row_count,
	           const export_func& f,
	           const std::string& sep_char = default_sep_char,
	           const std::string& quote_char = default_quote_char,
	           const std::string& header = std::string());

  public:
	size_t export_rows(PVRow step_count = 0);

	void cancel();
	void wait_finished();

  private:
	std::string _file_path;
	const PVCore::PVSelBitField& _sel;
	const PVCore::PVColumnIndexes& _column_indexes;
	PVRow _total_row_count = 0;
	export_func _f;
	const std::string _sep_char;
	const std::string _quote_char;
	PVCore::PVStreamingCompressor _compressor;
	PVRow _exported_row_count = 0;
};

} // namespace PVCore

#endif // __PVCORE_PVEXPORTER_H__

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
  private:
	static const std::unordered_map<size_t, std::pair<std::string, std::string>> _binaries;

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
	           PVRow step_count,
	           const export_func& f,
	           const std::string& sep_char = default_sep_char,
	           const std::string& quote_char = default_quote_char,
	           const std::string& header = std::string());

  public:
	void export_rows(size_t start_index);

	void set_step_count(PVRow step) { _step_count = step; }

	void cancel();
	void wait_finished();

  private:
	std::string _file_path;
	const PVCore::PVSelBitField& _sel;
	const PVCore::PVColumnIndexes& _column_indexes;
	PVRow _step_count;
	const std::string _sep_char;
	const std::string _quote_char;
	export_func _f;
	PVCore::PVStreamingCompressor _compressor;
};

} // namespace PVCore

#endif // __PVCORE_PVEXPORTER_H__

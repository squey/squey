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

#include <pvkernel/core/PVSelBitField.h> // for PVSelBitField

namespace PVCore
{
class PVSelBitField;

struct PVExportError : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

class PVExporterBase
{
  public:
	PVExporterBase(const std::string& file_path, const PVCore::PVSelBitField& sel)
	    : _file_path(file_path), _sel(sel)
	{
	}
	virtual ~PVExporterBase() {}

  public:
	void set_progress_max_func(const std::function<void(size_t max_progress)>& f_progress_max)
	{
		f_progress_max(_sel.bit_count());
	}
	void set_progress_func(const std::function<void(size_t current_progress)>& f_progress)
	{
		_f_progress = f_progress;
	}

  public:
	virtual void export_rows() = 0;

  public:
	virtual void cancel() { _canceled = true; }

  protected:
	std::string _file_path;
	const PVCore::PVSelBitField& _sel;
	bool _canceled = false;

	std::function<void(size_t current_progress)> _f_progress;
};

class PVCSVExporter : public PVExporterBase
{
  public:
	static const size_t STEP_COUNT;
	static const std::string default_sep_char;
	static const std::string default_quote_char;

  public:
	using export_func = std::function<std::string(
	    PVRow, const PVCore::PVColumnIndexes&, const std::string&, const std::string&)>;

  public:
	PVCSVExporter(const std::string& file_path,
	              const PVCore::PVSelBitField& sel,
	              const PVCore::PVColumnIndexes& column_indexes,
	              PVRow total_row_count,
	              const export_func& f,
	              const std::string& sep_char = default_sep_char,
	              const std::string& quote_char = default_quote_char,
	              const std::string& header = std::string());

  public:
	void export_rows();

  private:
	const PVCore::PVColumnIndexes& _column_indexes;
	size_t _total_row_count = 0;
	export_func _f;
	const std::string _sep_char;
	const std::string _quote_char;
	PVCore::PVStreamingCompressor _compressor;
};

} // namespace PVCore

#endif // __PVCORE_PVEXPORTER_H__

/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef __PVRUSH_PVEXPORTER_H__
#define __PVRUSH_PVEXPORTER_H__

#include <pvkernel/core/PVColumnIndexes.h> // for PVColumnIndexes
#include <pvkernel/core/PVStreamingCompressor.h>
#include <pvkernel/core/PVSelBitField.h> // for PVSelBitField
#include <pvkernel/core/PVArgument.h>

#include "pvbase/types.h" // for PVRow

#include <atomic>
#include <cstddef>    // for size_t
#include <fstream>    // for ostream
#include <functional> // for function
#include <string>     // for string
#include <thread>     // for std::thread
#include <unordered_map>

namespace PVRush
{
class PVSelBitField;

struct PVExportError : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

class PVExporterBase
{
  public:
	PVExporterBase(){};
	virtual ~PVExporterBase() {}

  public:
	void set_progress_max_func(const std::function<void(size_t max_progress)>& f_progress_max,
	                           size_t max_progress)
	{
		f_progress_max(max_progress);
	}
	void set_progress_func(const std::function<void(size_t current_progress)>& f_progress)
	{
		_f_progress = f_progress;
	}

  public:
	virtual void export_rows(const std::string& file_path, const PVCore::PVSelBitField& sel) = 0;

  public:
	virtual void cancel() { _canceled = true; }

  protected:
	bool _canceled = false;

	std::function<void(size_t current_progress)> _f_progress;
};

} // namespace PVRush

#endif // __PVRUSH_PVEXPORTER_H__

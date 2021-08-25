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

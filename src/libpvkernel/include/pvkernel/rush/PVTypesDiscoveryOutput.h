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

#ifndef __PVRUSH_PVTYPESDISCOVERYOUTPUT_H__
#define __PVRUSH_PVTYPESDISCOVERYOUTPUT_H__

#include <pvkernel/rush/PVControllerJob.h> // for PVControllerJob, etc
#include <pvkernel/rush/PVOutput.h>        // for PVOutput

#include <pvkernel/filter/PVFilterFunction.h> // for CLASS_FILTER_NONREG

#include <pvbase/types.h> // for PVRow
#include <pvbase/general.h>

#include <unordered_set>
#include <memory>

namespace pvcop
{
namespace types
{
class formatter_interface;
}
} // namespace pvcop

namespace PVRush
{

class PVTypesDiscoveryOutput : public PVRush::PVOutput
{
  public:
	using autodet_type_t =
	    std::vector<std::pair<std::pair<std::string, std::string>, std::vector<std::string>>>;
	using type_desc_t =
	    std::tuple<std::string, std::string, std::string>; /* type, type_format, name */

  public:
	// This is the output of a TBB pipeline
	// It takes a PVCore::PVTextChunk* as a parameter, and do whatever he wants with it
	// It *must* call PVChunk->free() in the end !!
	void operator()(PVCore::PVChunk* out) override;
	PVRow get_rows_count() override { return _row_count; }

  public:
	type_desc_t type_desc(PVCol col) const;
	static void append_time_formats(const std::unordered_set<std::string>& time_formats);
	static std::unordered_set<std::string> supported_time_formats();

  protected:
	void prepare_load(const PVRush::PVFormat&) override;
	void job_has_finished(const PVControllerJob::invalid_elements_t&) override;

	CLASS_FILTER_NONREG(PVNrawTypesDiscovery)

  private:
	using matching_formatters_t = std::vector<std::vector<bool>>;
	using types_desc_t = std::vector<type_desc_t>;

	autodet_type_t _types;
	std::vector<std::unique_ptr<pvcop::types::formatter_interface>> _formatters;
	types_desc_t _types_desc;
	std::vector<std::string> _names;
	matching_formatters_t _matching_formatters;
	size_t _column_count = 0;
	size_t _row_count = 0;
};
} // namespace PVRush

#endif // __PVRUSH_PVTYPESDISCOVERYOUTPUT_H__

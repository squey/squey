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

#ifndef INENDI_PVSOURCE_H
#define INENDI_PVSOURCE_H

#include <inendi/PVMapped.h> // for PVMapped

#include <pvkernel/rush/PVExtractor.h>         // for PVExtractor
#include <pvkernel/rush/PVFormat.h>            // for PVFormat
#include <pvkernel/rush/PVInputType.h>         // for PVInputType, etc
#include <pvkernel/rush/PVNraw.h>              // for PVNraw
#include <pvkernel/rush/PVSourceCreator.h>     // for PVSourceCreator_p, etc
#include <pvkernel/rush/PVSourceDescription.h> // for PVSourceDescription
#include <pvkernel/rush/PVControllerJob.h>     // for PVControllerJob_p

#include <pvkernel/core/PVDataTreeObject.h> // for PVDataTreeChild, etc

#include <pvbase/types.h> // for PVRow, PVCol

#include <QString>

#include <cstddef> // for size_t
#include <map>     // for map
#include <memory>  // for allocator, __shared_ptr
#include <string>  // for string, operator+, etc


namespace Inendi
{
class PVScene;
} // namespace Inendi
namespace Inendi
{
class PVView;
} // namespace Inendi
namespace PVCore
{
class PVSerializeObject;
} // namespace PVCore

namespace Inendi
{

class PVScene;

/**
 * \class PVSource
 */
class PVSource : public PVCore::PVDataTreeParent<PVMapped, PVSource>,
                 public PVCore::PVDataTreeChild<PVScene, PVSource>
{
  public:
	PVSource(Inendi::PVScene& scene,
	         PVRush::PVInputType::list_inputs_desc const& inputs,
	         PVRush::PVSourceCreator_p sc,
	         PVRush::PVFormat const& format);
	PVSource(PVScene& scene, const PVRush::PVSourceDescription& descr)
	    : PVSource(scene, descr.get_inputs(), descr.get_source_creator(), descr.get_format())
	{
	}
	~PVSource();

  public:
	void load_data() { wait_extract_end(extract(0)); }

	/* Functions */
	PVCol get_nraw_column_count() const;

	PVRush::PVNraw& get_rushnraw();
	const PVRush::PVNraw& get_rushnraw() const;

	std::string get_value(PVRow row, PVCol col) const;

	/**
	 * Check if the value encountered during import has failed
	 * to be converted correctly
	 *
	 * @return true if the conversion has failed
	 *         false otherwise
	 */
	bool is_valid(PVRow row, PVCol col) const;

	/**
	 * Check if a column contains at least one invalid value
	 */
	pvcop::db::INVALID_TYPE has_invalid(PVCol col) const;

	/**
	 * Return the number of row in the datastorage.
	 */
	PVRow get_row_count() const;

	/**
	 * Return number of correctly splitted row in the datastorage.
	 */
	PVRow get_valid_row_count() const;

	/**
	 * Start extraction of data for current source.
	 *
	 * @param skip_lines_count : Number of line to skip at the beginning of the file.
	 *
	 * @return : Pointer to the started job.
	 */
	PVRush::PVControllerJob_p extract(size_t skip_lines_count);
	void wait_extract_end(PVRush::PVControllerJob_p job);

	std::map<size_t, std::string> const& get_invalid_evts() const { return _inv_elts; }

	PVRush::PVInputType::list_inputs const& get_inputs() const { return _inputs; }

	PVRush::PVSourceCreator_p get_source_creator() const { return _src_plugin; }
	std::string get_name() const
	{
		return _src_plugin->supported_type_lib()->tab_name_of_inputs(_inputs).toStdString();
	}
	QString get_format_name() const { return get_format().get_format_name(); }
	QString get_window_name() const;
	QString get_tooltip() const;

	PVView* last_active_view() const { return _last_active_view; }

	PVView* current_view();
	PVView const* current_view() const;

	PVRush::PVFormat const& get_format() const { return _format; }
	PVRush::PVFormat const& get_original_format() const { return _original_format; }

	std::string get_serialize_description() const override { return "Source: " + get_name(); }

	/**
	 * Return the cumulate total size of all inputs.
	 *
	 * Unity is input defined. eg: octet for file, line for ElasticSearch, ...
	 * 0 means unknonwn.
	 */
	size_t max_size() const { return _extractor.max_size(); }

  public:
	inline void set_last_active_view(Inendi::PVView* view) { _last_active_view = view; }

  public:
	static PVSource& serialize_read(PVCore::PVSerializeObject& so, PVScene& parent);
	void serialize_write(PVCore::PVSerializeObject& so) const;

  private:
	std::string hash() const;

  private:
	PVView* _last_active_view = nullptr;

	PVRush::PVFormat
	    _original_format; //!< Format use to create the source (also contains metadata like colors)
	PVRush::PVFormat _format; //!< Original format with potential additional "input_name" column
	PVRush::PVNraw _nraw;     //!< NRaw data
	PVRush::PVInputType::list_inputs _inputs;
	PVRush::PVNrawOutput _output;

	PVRush::PVSourceCreator_p _src_plugin;
	// FIXME : The extracor is an attribute as we can't create an extractor from source (it would
	// required a move constructor)
	PVRush::PVExtractor _extractor;
	std::map<size_t, std::string> _inv_elts; //!< List of invalid elements sorted by line number.
};
} // namespace Inendi

#endif /* INENDI_PVSOURCE_H */

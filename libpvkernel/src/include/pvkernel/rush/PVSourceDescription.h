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

#ifndef PVSOURCEDESCRIPTION_H_
#define PVSOURCEDESCRIPTION_H_

#include <pvkernel/rush/PVFormat.h>        // for PVFormat
#include <pvkernel/rush/PVSourceCreator.h> // for PVSourceCreator_p
#include <pvkernel/rush/PVInputType.h>     // for PVInputType::list_inputs, etc

#include <pvkernel/core/PVSerializedSource.h>

namespace PVRush
{

class PVSourceDescription
{
  public:
	PVSourceDescription() : _inputs(), _source_creator_p(), _format() {}

	PVSourceDescription(PVRush::PVInputType::list_inputs inputs,
	                    PVRush::PVSourceCreator_p source_creator_p,
	                    PVRush::PVFormat format)
	    : _inputs(std::move(inputs))
	    , _source_creator_p(std::move(source_creator_p))
	    , _format(std::move(format))
	{
	}

	explicit PVSourceDescription(PVCore::PVSerializedSource const& s)
	    : _source_creator_p(LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name(
	          QString::fromStdString(s.sc_name)))
	    , _format(QString::fromStdString(s.format_name), QString::fromStdString(s.format_path))
	{
		PVRush::PVInputType_p input_type_p = _source_creator_p->supported_type_lib();
		for (auto const& p : s.input_desc) {
			_inputs << input_type_p->load_input_from_string(p, s.input_desc.size() > 1);
		}
	}

	bool operator==(const PVSourceDescription& other) const;
	bool operator!=(const PVSourceDescription& other) const { return !operator==(other); }

	void set_inputs(const PVRush::PVInputType::list_inputs inputs) { _inputs = inputs; }
	void set_source_creator(PVRush::PVSourceCreator_p source_creator_p)
	{
		_source_creator_p = source_creator_p;
	}
	void set_format(PVRush::PVFormat format) { _format = format; }

	const PVRush::PVInputType::list_inputs& get_inputs() const { return _inputs; }
	PVRush::PVSourceCreator_p get_source_creator() const { return _source_creator_p; }
	PVRush::PVFormat get_format() const { return _format; }

  private:
	PVRush::PVInputType::list_inputs _inputs;
	PVRush::PVSourceCreator_p _source_creator_p;
	PVRush::PVFormat _format;
};
} // namespace PVRush

#endif /* PVSOURCEDESCRIPTION_H_ */

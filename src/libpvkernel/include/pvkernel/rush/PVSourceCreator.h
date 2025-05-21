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

#ifndef SQUEY_PVSOURCECREATOR_H
#define SQUEY_PVSOURCECREATOR_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVInputType.h>

#include <memory>

#ifndef BUILD_SQUEY
template class PVCore::PVClassLibrary<PVRush::PVInputType>;
#endif

namespace PVRush
{

class PVRawSourceBase;

class PVSourceCreator : public PVCore::PVRegistrableClass<PVSourceCreator>
{
  public:
	typedef PVRush::PVRawSourceBase source_t;
	typedef std::shared_ptr<source_t> source_p;
	typedef std::shared_ptr<PVSourceCreator> p_type;

  public:
	virtual ~PVSourceCreator() = default;

  public:
	virtual source_p create_source_from_input(PVInputDescription_p input) const = 0;
	virtual QString supported_type() const = 0;
	PVInputType_p supported_type_lib()
	{
		QString name = supported_type();
		PVInputType_p type_lib = LIB_CLASS(PVInputType)::get().get_class_by_name(name);
		return type_lib->clone<PVInputType>();
	}
	virtual QString name() const = 0;
	virtual bool custom_multi_inputs() const { return false; }
};

typedef PVSourceCreator::p_type PVSourceCreator_p;
} // namespace PVRush

#endif

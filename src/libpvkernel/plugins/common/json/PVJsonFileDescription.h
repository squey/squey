/* * MIT License
 *
 * © Squey, 2026
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

#ifndef __RUSH_PVJSONFILEDESCRIPTION_H__
#define __RUSH_PVJSONFILEDESCRIPTION_H__

#include <pvkernel/rush/PVFileListDescription.h>

#include "PVJsonSchema.h"

#include <memory>
#include <utility>

namespace PVRush
{

/**
 * JSON files, along with the schema that was found in them to make their format, which spares
 * their import another reading of the files.
 */
class PVJsonFileDescription : public PVFileListDescription
{
  public:
	using PVFileListDescription::PVFileListDescription;

  public:
	/** nullptr until set_schema() is called. */
	std::shared_ptr<const PVJsonSchema> schema() const { return _schema; }
	void set_schema(std::shared_ptr<const PVJsonSchema> schema) { _schema = std::move(schema); }

  private:
	std::shared_ptr<const PVJsonSchema> _schema;
};

} // namespace PVRush

#endif // __RUSH_PVJSONFILEDESCRIPTION_H__

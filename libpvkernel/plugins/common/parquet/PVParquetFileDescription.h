/* * MIT License
 *
 * © Squey, 2024
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

#ifndef __PVPARQUETDESCRIPTION_H__
#define __PVPARQUETDESCRIPTION_H__

#include <pvkernel/rush/PVFileDescription.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/pointer.h>

namespace PVRush
{

class PVParquetFileDescription : public PVFileDescription
{
  public:
	PVParquetFileDescription(
		QStringList const& paths
	)
	    : PVFileDescription(paths.front(), paths.size() > 1 /*multi_inputs*/)
		, _paths(paths)
	{
	}

  public:
  	const QStringList& paths() const { return _paths; }

	void disable_multi_inputs(bool disabled) { _multi_inputs = not disabled; }

  private:
  	QStringList _paths;

  public:
	void serialize_write(PVCore::PVSerializeObject& so) const override
	{
		so.set_current_status("Saving source file information...");
		pvlogger::warn() << _paths.join("\n").toStdString() << std::endl;
		so.attribute_write("files_path", _paths);
	}

	static std::unique_ptr<PVInputDescription> serialize_read(PVCore::PVSerializeObject& so)
	{
		so.set_current_status("Loading source file information...");
		QStringList paths = so.attribute_read<QStringList>("files_path");

		// File exists, continue with it
		for (const QString& path : paths) {
			if (not QFileInfo(path).isReadable()) {
				throw PVCore::PVSerializeReparaibleFileError(
					"Source file: '" + path.toStdString() + "' can't be found",
					so.get_logical_path().toStdString(), path.toStdString());
			}
		}

		return std::unique_ptr<PVInputDescription>(new PVParquetFileDescription(paths));
	}
};

} // namespace PVRush

#endif // __PVPARQUETDESCRIPTION_H__

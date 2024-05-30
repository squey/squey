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

#ifndef __PVRUSH_PVPARQUETEXPORTER_H__
#define __PVRUSH_PVPARQUETEXPORTER_H__

#include <QDesktopServices>
#include <QCheckBox>
#include <QUrl>

#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVNraw.h>
#include <pvkernel/rush/PVExporter.h>

#include <arrow/util/type_fwd.h>

namespace PVRush
{

class PVParquetExporter : public PVExporterBase
{
  public:
	PVParquetExporter(const PVRush::PVInputType::list_inputs& inputs, PVRush::PVNraw const& nraw);
	virtual ~PVParquetExporter(){};

  public:
	void export_rows(const std::string& out_path, const PVCore::PVSelBitField& sel) override;

  public:
	void set_compression_codec(arrow::Compression::type codec) { _compression_codec = codec; }
	arrow::Compression::type compression_codec() const { return _compression_codec; }

  protected:
	const PVRush::PVInputType::list_inputs& _inputs;
	PVRush::PVNraw const& _nraw;
	std::vector<std::pair<std::string, size_t>> _input_parquet;
	pvcop::db::indexes _sorted_indexes;
	arrow::Compression::type _compression_codec = arrow::Compression::UNCOMPRESSED;
};

} // namespace PVRush

#endif // __PVRUSH_PVPARQUETEXPORTER_H__

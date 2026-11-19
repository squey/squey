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

#ifndef __PVRUSH_PVPARQUETNRAWEXPORTER_H__
#define __PVRUSH_PVPARQUETNRAWEXPORTER_H__

#include <pvkernel/rush/PVNraw.h>

#include "PVParquetExporterBase.h"

#include <QStringList>

#include <string>

namespace PVRush
{

/**
 * Exports the rows of the NRAW of a source of any type.
 *
 * Numbers, times and durations keep their types, and the other values come out as the text the
 * CSV export writes. The values that were invalid on import come out as nulls, but for those
 * written as text, which keep the text they were imported from.
 */
class PVParquetNrawExporter : public PVParquetExporterBase
{
  public:
	/**
	 * The columns of @p nraw are named @p column_names, in their order. @p nraw has to outlive
	 * the exporter.
	 */
	PVParquetNrawExporter(PVRush::PVNraw const& nraw, QStringList column_names);

  public:
	/**
	 * Writes all of the columns of the NRAW, unless set_columns() chose some. Columns of the
	 * same name get a number, as readers find the columns of a Parquet file by their names.
	 */
	void export_rows(const std::string& path, const PVCore::PVSelBitField& sel) override;

  private:
	PVRush::PVNraw const& _nraw;
	QStringList _column_names;
};

} // namespace PVRush

#endif // __PVRUSH_PVPARQUETNRAWEXPORTER_H__

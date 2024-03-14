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

#ifndef __PVINPUTTYPEPARQUET_H__
#define __PVINPUTTYPEPARQUET_H__

#include <pvkernel/rush/PVInputType.h>
#include "../common/parquet/PVParquetFileDescription.h"

#include <QString>
#include <QStringList>
#include <QIcon>
#include <QCursor>

namespace PVRush
{

class PVInputTypeParquet : public PVInputTypeDesc<PVParquetFileDescription>
{
  public:
	PVInputTypeParquet();

  public:
	bool createWidget(hash_formats& formats,
	                  list_inputs& inputs,
	                  QString& format,
	                  PVCore::PVArgumentList& args_ext,
	                  QWidget* parent = nullptr) const override;
	QString name() const override;
	QString human_name() const override;
	QString human_name_serialize() const override;
	QString internal_name() const override;
	QString menu_input_name() const override;
	QString tab_name_of_inputs(list_inputs const& in) const override;
	QKeySequence menu_shortcut() const override;
	bool get_custom_formats(PVInputDescription_p /*in*/, hash_formats& /*formats*/) const override { return false; }

	QIcon icon() const override { return QIcon(":/parquet_icon"); }
	QCursor cursor() const override { return QCursor(Qt::PointingHandCursor); }

  protected:
	mutable QStringList _tmp_dir_to_delete;
	int _limit_nfds;

  protected:
	CLASS_REGISTRABLE_NOCOPY(PVInputTypeParquet)
};
} // namespace PVRush

#endif

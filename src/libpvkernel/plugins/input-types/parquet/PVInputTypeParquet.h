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

#include "parquet/PVParquetExporter.h"
#include "parquet/PVParquetExporterWidget.h"
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
	bool create_widget(
	    hash_formats& formats,
        list_inputs& inputs,
        QString& format,
        PVCore::PVArgumentList& args_ext,
        QWidget* parent = nullptr
	) const override;

	bool create_widget_with_input_files(
	    const QStringList& input_paths,
	    hash_formats& formats,
        list_inputs& inputs,
        QString& format,
        PVCore::PVArgumentList& args_ext,
        QWidget* parent = nullptr
	) const override;

	bool create_source_description_params(const QString& params_json, list_inputs& inputs, PVFormat& format) const override;

	/* exporter */
	std::unique_ptr<PVRush::PVExporterBase>
	create_exporter(const list_inputs& inputs, PVRush::PVNraw const& nraw) const override;
	PVWidgets::PVExporterWidgetInterface*
	create_exporter_widget(const list_inputs& inputs, PVRush::PVNraw const& nraw) const override;
	QString get_exporter_filter_string(const list_inputs& inputs) const override;

	QString name() const override;
	QString human_name() const override;
	QString human_name_serialize() const override;
	QString internal_name() const override;
	QStringList get_supported_extensions() const override;
	QString menu_input_name() const override;
	QString tab_name_of_inputs(list_inputs const& in) const override;
	bool get_custom_formats(PVInputDescription_p /*in*/, hash_formats& /*formats*/) const override { return false; }

	PVModdedIcon icon() const override { return PVModdedIcon("parquet"); }
	QCursor cursor() const override { return QCursor(Qt::PointingHandCursor); }

  private:
	bool create_source_description_params(const QStringList& paths, list_inputs& inputs, PVFormat& format) const;

  protected:
	mutable QStringList _tmp_dir_to_delete;
	int _limit_nfds;

  protected:
	CLASS_REGISTRABLE_NOCOPY(PVInputTypeParquet)
};
} // namespace PVRush

#endif

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

#include "parquet/PVParquetExporterWidget.h"
#include "parquet/PVParquetExporter.h"
#include "../../common/parquet/PVParquetAPI.h"

#include <QComboBox>
#include <QLabel>
#include <QVBoxLayout>

#include <arrow/io/api.h>
#include <arrow/util/compression.h>

PVWidgets::PVParquetExporterWidget::PVParquetExporterWidget(const PVRush::PVInputType::list_inputs& inputs, PVRush::PVNraw const& nraw)
    : _exporter(new PVRush::PVParquetExporter(inputs, nraw))
{
    QHBoxLayout* compression_codecs_layout = new QHBoxLayout;

    QLabel* compression_codecs_label = new QLabel("Compression codec:");
    QComboBox* compression_codecs_combobox = new QComboBox();

    compression_codecs_layout->addWidget(compression_codecs_label);
    compression_codecs_layout->addWidget(compression_codecs_combobox);
    compression_codecs_layout->addStretch();

    // Dynamically add all compression codecs supported by Apache Arrow
    for (size_t i = 0; i < std::numeric_limits<uint8_t>::max(); i++) {
        arrow::Compression::type codec = (arrow::Compression::type)i;
        if (arrow::util::Codec::IsAvailable(codec)) {
            compression_codecs_combobox->addItem(QString::fromStdString(arrow::util::Codec::GetCodecAsString(codec)), QVariant((int)codec));
        }
    }

    QObject::connect(compression_codecs_combobox, &QComboBox::currentIndexChanged, [this,compression_codecs_combobox](int /*index*/) {
        PVRush::PVParquetExporter* parquet_exporter = static_cast<PVRush::PVParquetExporter*>(_exporter.get());
        parquet_exporter->set_compression_codec((arrow::Compression::type)compression_codecs_combobox->currentData(Qt::UserRole).toInt()); 
    });

    // Set the default compression codec the same as the one used on the input file
    try {
        PVRush::PVParquetFileDescription* input_desc = dynamic_cast<PVRush::PVParquetFileDescription*>(inputs.front().get());
        PVRush::PVParquetAPI api(input_desc);
        arrow::Compression::type used_codec = api.arrow_reader()->parquet_reader()->metadata()->RowGroup(0)->ColumnChunk(0)->compression();
        int used_codec_index = compression_codecs_combobox->findData(QVariant((int) used_codec));
        if (used_codec_index != -1) {
            compression_codecs_combobox->setCurrentIndex(used_codec_index);
        }
    }
    catch (...) {
        // let the exception raise during the export
    }

    setLayout(compression_codecs_layout);
}


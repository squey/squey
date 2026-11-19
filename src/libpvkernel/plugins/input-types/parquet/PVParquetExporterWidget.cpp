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

#include <QComboBox>
#include <QLabel>
#include <QVBoxLayout>

#include <arrow/util/compression.h>

#include <algorithm>
#include <utility>

PVWidgets::PVParquetExporterWidget::PVParquetExporterWidget(std::unique_ptr<PVRush::PVParquetExporterBase> exporter, arrow::Compression::type codec)
    : _exporter(std::move(exporter))
{
    QHBoxLayout* compression_codecs_layout = new QHBoxLayout;

    QLabel* compression_codecs_label = new QLabel("Compression codec:");
    QComboBox* compression_codecs_combobox = new QComboBox();

    compression_codecs_layout->addWidget(compression_codecs_label);
    compression_codecs_layout->addWidget(compression_codecs_combobox);
    compression_codecs_layout->addStretch();

    for (arrow::Compression::type supported_codec : PVRush::PVParquetExporterBase::compression_codecs()) {
        compression_codecs_combobox->addItem(QString::fromStdString(arrow::util::Codec::GetCodecAsString(supported_codec)), QVariant((int)supported_codec));
    }

    const auto select_codec = [this, compression_codecs_combobox]() {
        _exporter->set_compression_codec((arrow::Compression::type)compression_codecs_combobox->currentData(Qt::UserRole).toInt());
    };
    compression_codecs_combobox->setCurrentIndex(std::max(compression_codecs_combobox->findData(QVariant((int)codec)), 0));
    select_codec();
    QObject::connect(compression_codecs_combobox, &QComboBox::currentIndexChanged, select_codec);

    setLayout(compression_codecs_layout);
}


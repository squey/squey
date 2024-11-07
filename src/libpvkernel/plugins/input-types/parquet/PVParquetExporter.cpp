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

#include "PVParquetExporter.h"
#include "../../common/parquet/PVParquetAPI.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/io/file.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/compute/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

template <typename TYPE>
static std::shared_ptr<arrow::Array> array_from_indexes(
    const std::shared_ptr<arrow::DataType>& type,
    const pvcop::core::array<uint32_t>& indexes
)
{
    std::unique_ptr<arrow::ArrayBuilder> builder_ptr;
    PARQUET_THROW_NOT_OK(MakeBuilder(arrow::default_memory_pool(), type, &builder_ptr));
    auto& builder = dynamic_cast<typename arrow::TypeTraits<TYPE>::BuilderType&>(*builder_ptr);
    for (size_t i = 0; i < indexes.size(); ++i) {
        auto status = builder.Append(indexes[i]);
    }
    std::shared_ptr<arrow::Array> out;
    auto status = builder.Finish(&out);
    return out;
}

PVRush::PVParquetExporter::PVParquetExporter(const PVRush::PVInputType::list_inputs& inputs, PVRush::PVNraw const& nraw)
    : _inputs(inputs), _nraw(nraw)
{
}

void PVRush::PVParquetExporter::export_rows(const std::string & out_path, const PVCore::PVSelBitField & sel)
{
    PVRush::PVParquetFileDescription* input_desc = dynamic_cast<PVRush::PVParquetFileDescription*>(_inputs.front().get());

    const auto& paths = input_desc->paths();
    if (paths.contains(QString::fromStdString(out_path))) {
        throw std::runtime_error("Cannot overwrite an input file during export.");
        return;
    }

    std::shared_ptr<parquet::WriterProperties> parquet_props = parquet::WriterProperties::Builder().compression(compression_codec())->build();
    std::shared_ptr<parquet::ArrowWriterProperties> arrow_props = parquet::ArrowWriterProperties::Builder().store_schema()->build();
    std::shared_ptr<arrow::fs::FileSystem> fs = arrow::fs::FileSystemFromUri("file:///").ValueOrDie();
    auto output_stream = fs->OpenOutputStream(out_path).ValueOrDie();
    std::shared_ptr<arrow::io::FileOutputStream> file_output_stream = std::dynamic_pointer_cast<arrow::io::FileOutputStream>(output_stream);
    std::unique_ptr<parquet::arrow::FileWriter> writer;

    const PVCore::PVSelBitField::pvcop_selection_t& selected_rows = (const PVCore::PVSelBitField::pvcop_selection_t&) sel;

    size_t exported_rows_count = 0;
    const size_t rows_count_to_export = sel.bit_count();
    PVRush::PVParquetAPI api(input_desc);
    size_t selection_current_index = 0;
    api.visit_files([&](){
        if (writer == nullptr) {
            std::shared_ptr<arrow::Schema> schema;
            arrow::Status status = api.arrow_reader()->GetSchema(&schema);
            writer = parquet::arrow::FileWriter::Open(*schema, arrow::default_memory_pool(), output_stream, parquet_props, arrow_props).ValueOrDie();
        }

        std::unique_ptr<::arrow::RecordBatchReader> recordbatch_reader;
        arrow::Status status = api.arrow_reader()->GetRecordBatchReader(&recordbatch_reader);

        size_t total_rows = api.arrow_reader()->parquet_reader()->metadata()->num_rows();
        size_t batch_current_index = 0;
        while (batch_current_index < total_rows and exported_rows_count < rows_count_to_export) {
            std::shared_ptr<arrow::RecordBatch> batch = recordbatch_reader->Next().ValueOrDie();

            size_t selected_record_batch_indexes_size = pvcop::core::algo::bit_count(selected_rows, selection_current_index + batch_current_index, selection_current_index + batch_current_index + batch->num_rows() -1);

            pvcop::db::array selected_record_batch_indexes("number_uint32", selected_record_batch_indexes_size);
            auto& selected_record_batch_indexes_core = selected_record_batch_indexes.to_core_array<uint32_t>();
            size_t index = 0;
            sel.visit_selected_lines([&](int row_id) {
                selected_record_batch_indexes_core[index++] = row_id - (selection_current_index + batch_current_index);

            }, selection_current_index + batch_current_index + batch->num_rows(), selection_current_index + batch_current_index);

            if (selected_record_batch_indexes_core.size() > 0) {

                // filter record batch
                std::shared_ptr<arrow::Array> indexes = array_from_indexes<arrow::Int32Type>(arrow::int32(), selected_record_batch_indexes_core);
                arrow::Datum result = arrow::compute::Take(batch, indexes).ValueOrDie();
                auto status = writer->WriteRecordBatch(*result.record_batch());

                if (_f_progress) {
                    _f_progress(exported_rows_count += indexes->length());
                }
                if (_canceled) {
                    PARQUET_THROW_NOT_OK(writer->Close());
                    std::remove(out_path.c_str());
                    _canceled = false;
                    return;
                }
            }

            batch_current_index += batch->num_rows();
        }
        selection_current_index += batch_current_index;
    });

    PARQUET_THROW_NOT_OK(writer->Close());
}
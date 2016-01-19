/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVSelBitField.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>

#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/rush/PVNraw.h>
#include <pvkernel/rush/PVNrawException.h>
#include <pvkernel/rush/PVUtils.h>

#include <pvcop/collector.h>
#include <pvcop/sink.h>

#include <tbb/tbb_allocator.h>
#include <tbb/tick_count.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>

#include <iostream>

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include <QDir>
#include <QFileInfo>

#define DEFAULT_LINE_SIZE 100
#define MAX_SIZE_RESERVE (size_t(1024*1024*1024u)) // 1GB

const QString PVRush::PVNraw::config_nraw_tmp = "pvkernel/nraw_tmp";
const QString PVRush::PVNraw::default_tmp_path = QDir::tempPath() + "/inendi";
const QString PVRush::PVNraw::nraw_tmp_pattern = "nraw-XXXXXX";
const QString PVRush::PVNraw::nraw_tmp_name_regexp = "nraw-??????";
const QString PVRush::PVNraw::default_sep_char = ",";
const QString PVRush::PVNraw::default_quote_char = "\"";


PVRush::PVNraw::PVNraw()
{
	_real_nrows = 0;

	UErrorCode status = U_ZERO_ERROR;
	_ucnv = ucnv_open("UTF8", &status);
}

PVRush::PVNraw::~PVNraw()
{
	ucnv_close(_ucnv);
	_real_nrows = 0;
	delete _format;
	delete _collector;
	delete _collection;
}

void PVRush::PVNraw::reserve(PVRow const nrows, PVCol const ncols)
{
	// Generate random path
	QString nraw_dir_base = PVRush::PVNrawCacheManager::nraw_dir() + QDir::separator() + nraw_tmp_pattern;
	QByteArray nstr = nraw_dir_base.toLocal8Bit();
	if (mkdtemp(nstr.data()) == nullptr) {
		throw PVNrawException(QObject::tr("unable to create temporary directory ") + nraw_dir_base);
	}

	// Create collector and format
	// FIXME : Memory leak inside
	std::string const collector_path = nstr.constData();
	_format = new pvcop::format(get_format()->get_storage_format());
	_collector = new pvcop::collector(collector_path.data(), *_format);

	// Define maximum number of row;
	if(nrows == 0) {
		_max_nrows = INENDI_LINES_MAX;
	} else {
		_max_nrows = nrows;
	}
}

bool PVRush::PVNraw::add_chunk_utf16(PVCore::PVChunk const& chunk)
{
	if (_real_nrows == _max_nrows) {
		// the whole chunk can be skipped as we extracted enough data.
		return false;
	}

	const size_t column_count = _format->column_count();

	// Write all elements of the chunk in the final nraw
	PVCore::list_elts const& elts = chunk.c_elements();

	// Use the sink to write data from RAM to HDD
	pvcop::sink snk(*_collector, *_format);

	std::vector<pvcop::sink::field_t> pvcop_fields;
	pvcop_fields.reserve(elts.size() *  column_count);

	// Count number of extracted line. It is not the same as the number of elements as some of them
	// may be invalid or empty or we may skip the end when enough data is extracted.
	PVRow local_row = 0;
	for (PVCore::PVElement* elt: elts) {

		PVCore::PVElement& e = *elt;
		if (!e.valid()) {
			continue;
		}

		PVCore::list_fields const& fields = e.c_fields();
		if (fields.size() == 0) {
			continue;
		}

		if (_real_nrows == _max_nrows) {
			/* we have enough events, skips the others. As the
			 * chunk has been partially saved, the current chunked
			 * index has to be saved by the caller (PVNrawOutput).
			 */
			return true;
		}

		for (PVCore::PVField const& field :fields) {
			// TODO: make the whole process in utf8.. !
			// Convert field to UT8
			std::unique_ptr<char> tmp_buf(new char[field.size()]);
			UErrorCode err = U_ZERO_ERROR;
			size_t size_utf8 = ucnv_fromUChars(_ucnv, tmp_buf.get(), field.size(), (const UChar*) field.begin(), field.size()/sizeof(UChar), &err);
			if (!U_SUCCESS(err)) {
				PVLOG_WARN("Unable to convert a field to UTF8! Field is ignored..\n");
				continue;
			}

			// Save the field
			pvcop_fields.emplace_back(pvcop::sink::field_t{std::move(tmp_buf), size_utf8});
		}
		local_row++;
	}


	if (not snk.write_chunk_by_row(_real_nrows, elts.size(), pvcop_fields.data())) {
		PVLOG_WARN("Unable to write chunk to disk..\n");
	}

	_real_nrows += local_row;

	return true;
}

// Function call once import is done
// FIXME : It has to be rename
void PVRush::PVNraw::fit_to_content()
{
	if (_real_nrows > INENDI_LINES_MAX) {
		_real_nrows = INENDI_LINES_MAX;
	}

	// Close collector
	if (not _collector->close()) {
		PVLOG_ERROR("Error when closing collector..\n");
	}
	// FIXME : memory leak inside
	_collection = new pvcop::collection(_collector->rootdir());

	delete _collector;
	_collector = nullptr;
}

QString PVRush::PVNraw::get_value(PVRow row, PVCol col, bool* complete /*= nullptr*/) const
{
	assert(row < get_number_rows());
	assert(col < get_number_cols());
	if (complete) {
		*complete = true;
	}
	pvcop::db::array column = _collection->column(col);

	if (not column) {
		PVLOG_ERROR("Error when accessing column..\n");
		return {};
	}

	return column.at(row).c_str();
}

// FIXME : Should not return values.
bool PVRush::PVNraw::load_from_disk(const std::string& nraw_folder, PVCol ncols)
{
	_collection = new pvcop::collection(nraw_folder);
	_real_nrows = _collection->row_count();

	return true;
}

void PVRush::PVNraw::dump_csv()
{
	for (PVRow i = 0; i < get_number_rows(); i++) {
		std::cout << qPrintable(export_line(i)) << std::endl;
	}
}

void PVRush::PVNraw::dump_csv(const QString& file_path)
{
	FILE* file = fopen(qPrintable(file_path), "w");
	for (PVRow i = 0; i < get_number_rows(); i++) {
		const std::string& csv_line = export_line(i).toStdString();
		fwrite(csv_line.c_str(), csv_line.length(), 1, file);
		fputc('\n', file);
	}
	fclose(file);
}

QString PVRush::PVNraw::export_line(
	PVRow idx,
	PVCore::PVColumnIndexes col_indexes /* = PVCore::PVColumnIndexes() */,
	const QString sep_char /* = default_sep_char */,
	const QString quote_char /* = default_quote_char */
) const
{
	static QString escaped_quote("\\" + quote_char);

	size_t column_count = col_indexes.size() ? col_indexes.size() : get_number_cols();
	if (col_indexes.size() == 0) {
		for (size_t i = 0; i < column_count; i++) {
			col_indexes.push_back(i);
		}
	}

	QStringList fields;
	fields.reserve(column_count);
	for (size_t i = 0; i < column_count; i++) {
		fields.append("");
	}
	tbb::parallel_for(
		tbb::blocked_range<size_t>(tbb::blocked_range<size_t>(0, column_count, 1)),
		[&](tbb::blocked_range<size_t> const& range) {

			for (size_t c = range.begin(); c < range.end(); c++) {
				size_t col = col_indexes[c];

				assert(idx < get_number_rows());
				QString v(at(idx, col));

				PVRush::PVUtils::safe_export(v, sep_char, quote_char);

				fields[c] = v;
			}
		}
	), tbb::simple_partitioner();

	size_t line_length = 0;
	for (size_t i = 0; i < column_count; i++) {
		line_length += fields[i].size();
	}

	QString line = fields.join(sep_char);

	return line;
}

void PVRush::PVNraw::export_lines(
	QTextStream& stream,
	const PVCore::PVSelBitField& sel,
	const PVCore::PVColumnIndexes& col_indexes,
	size_t start_index,
	size_t step_count,
	const QString sep_char /* = default_sep_char */,
	const QString quote_char /* = default_quote_char */
) const
{
#ifndef NDEBUG
	PVCol ncols = get_number_cols();
	assert(ncols > 0);
#endif

	PVRow nrows_counter = 0;

	PVCore::PVColumnIndexes cols = col_indexes;
	if (cols.size() == 0) {
		for (PVCol i = 0; i < get_number_cols(); i++) {
			cols.push_back(i);
		}
	}

	for (PVRow line_index = start_index; line_index < start_index + step_count; line_index++) {
		if (!sel.get_line_fast(line_index)) {
			continue;
		}
		if (nrows_counter == step_count) {
			return;
		}

		QString line = export_line(line_index, cols, sep_char, quote_char);
		stream << line << QString("\n");

		nrows_counter++;
	}
}

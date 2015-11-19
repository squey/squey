/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/rush/PVNrawCacheManager.h>

#include <pvkernel/rush/PVNraw.h>
#include <pvkernel/rush/PVNrawException.h>
#include <pvkernel/rush/PVUtils.h>

#include <pvcop/collector.h>
#include <pvcop/sink.h>

#include <tbb/tick_count.h>

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


PVRush::PVNraw::PVNraw():
	_tmp_conv_buf(nullptr)
{
	_real_nrows = 0;

	UErrorCode status = U_ZERO_ERROR;
	_ucnv = ucnv_open("UTF8", &status);
}

PVRush::PVNraw::~PVNraw()
{
	ucnv_close(_ucnv);
	clear();
	delete _format;
}

void PVRush::PVNraw::reserve(PVRow const nrows, PVCol const ncols)
{
	// Generate random path
	QString nraw_dir_base = PVRush::PVNrawCacheManager::nraw_dir() + QDir::separator() + nraw_tmp_pattern;
	QByteArray nstr = nraw_dir_base.toLocal8Bit();
	if (mkdtemp(nstr.data()) == nullptr) {
		throw PVNrawException(QObject::tr("unable to create temporary directory ") + nraw_dir_base);
	}

	_backend.init(nstr.constData(), ncols);

	//const char* collector_path = (std::string(nstr.data()) + "_pvcop").c_str();
	const char* collector_path = "/srv/tmp-picviz/pvcop_nraw";
	delete _format;
	_format = new pvcop::format(get_format()->get_storage_format());
	_collector = new pvcop::collector(collector_path, *_format);

	if (not _collector) {
		PVLOG_ERROR("Collector failed to initialize properly\n");
	}

	if(nrows == 0) {
		_max_nrows = INENDI_LINES_MAX;
	} else {
		_max_nrows = nrows;
	}
}

void PVRush::PVNraw::clear()
{
	if (_tmp_conv_buf) {
		tbb::scalable_allocator<char>().deallocate(_tmp_conv_buf, _tmp_conv_buf_size);
	}
	_real_nrows = 0;
	_backend.clear_and_remove();
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

void PVRush::PVNraw::reserve_tmp_buf(size_t n)
{
	if (_tmp_conv_buf) {
		if (_tmp_conv_buf_size < n) {
			tbb::scalable_allocator<char>().deallocate(_tmp_conv_buf, _tmp_conv_buf_size);
			_tmp_conv_buf = tbb::scalable_allocator<char>().allocate(n);
			_tmp_conv_buf_size = n;
		}
	}
	else {
		_tmp_conv_buf = tbb::scalable_allocator<char>().allocate(n);
		_tmp_conv_buf_size = n;
	}
}

#define CHUNK_BY_COLUMN 0

bool PVRush::PVNraw::add_chunk_utf16(PVCore::PVChunk const& chunk)
{
	if (_real_nrows == _max_nrows) {
		// the whole chunk can be skipped
		return false;
	}

	const size_t column_count = _format->column_count();

	// Write all elements of the chunk in the final nraw
	PVCore::list_elts const& elts = chunk.c_elements();
	PVCore::list_elts::const_iterator it_elt;
	PVCore::list_fields::const_iterator it_field;

	pvcop::sink snk(*_collector, *_format);

	pvcop::sink::field_t* pvcop_fields = tbb::scalable_allocator<pvcop::sink::field_t>().allocate(elts.size() *  column_count);

	char* tmp_conv_buf[column_count][elts.size()];

	UErrorCode err = U_ZERO_ERROR;
	PVRow local_row = 0;
	for (it_elt = elts.begin(); it_elt != elts.end(); it_elt++) {
		PVCore::PVElement& e = *(*it_elt);
		if (!e.valid())
			continue;
		PVCore::list_fields const& fields = e.c_fields();
		if (fields.size() == 0)
			continue;

		if (_real_nrows == _max_nrows) {
			/* we have enough events, skips the others. As the
			 * chunk has been partially saved, the current chunked
			 * index has to be saved by the caller (PVNrawOutput).
			 */
			return true;
		}

		PVCol col = 0;
		for (it_field = fields.begin(); it_field != fields.end(); it_field++) {
			// Convert to UTF8
			// TODO: make the whole process in utf8.. !
			PVCore::PVField const& field = *it_field;
			char* tmp_buf = tmp_conv_buf[col][local_row] = tbb::scalable_allocator<char>().allocate(field.size());
			size_t size_utf8 = ucnv_fromUChars(_ucnv, tmp_buf, field.size(), (const UChar*) field.begin(), field.size()/sizeof(UChar), &err);
			if (!U_SUCCESS(err)) {
				PVLOG_WARN("Unable to convert field %d to UTF8! Field is ignored..\n", col);
				continue;
			}

			_backend.add(col, tmp_buf, size_utf8);

#if CHUNK_BY_COLUMN
			new (pvcop_fields + col * elts.size() + local_row) pvcop::sink::field_t(tmp_buf, size_utf8);

#else
			new (pvcop_fields + local_row * column_count + col) pvcop::sink::field_t(tmp_buf, size_utf8);
#endif

			col++;
		}
		local_row++;
	}


#if CHUNK_BY_COLUMN
	if (not snk.write_chunk_by_column(_real_nrows, elts.size(), pvcop_fields)) {
#else
	if (not snk.write_chunk_by_row(_real_nrows, elts.size(), pvcop_fields)) {
#endif
		PVLOG_WARN("Unable to write chunk to disk..\n");
	}

	_real_nrows += local_row;

	size_t row = 0;
	for (it_elt = elts.begin(); it_elt != elts.end(); it_elt++) {
		PVCore::PVElement& e = *(*it_elt);
		if (!e.valid())
			continue;
		PVCore::list_fields const& fields = e.c_fields();
		if (fields.size() == 0)
			continue;
		size_t col = 0;
		for (it_field = fields.begin(); it_field != fields.end(); it_field++) {
			PVCore::PVField const& field = *it_field;
			tbb::scalable_allocator<char>().deallocate(tmp_conv_buf[col++][row], field.size());
		}
		row++;
	}

	tbb::scalable_allocator<pvcop::sink::field_t>().deallocate(pvcop_fields, elts.size() *  column_count);

	return true;
}

void PVRush::PVNraw::fit_to_content()
{
	_backend.flush();
	if (_real_nrows > INENDI_LINES_MAX) {
		_real_nrows = INENDI_LINES_MAX;
	}

	// Close collector
	if (not _collector->close()) {
		PVLOG_ERROR("Error when closing collector..\n");
	}
	delete _collector;
	_collector = nullptr;

	// Open collection
	//_collection = new pvcop::collection((std::string(_backend.get_nraw_folder().c_str()) + "_pvcop").c_str());
	_collection = new pvcop::collection("/srv/tmp-picviz/pvcop_nraw");
	if (_collection) {
		PVLOG_ERROR("Error when opening collector..\n");
	}
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

bool PVRush::PVNraw::load_from_disk(const std::string& nraw_folder, PVCol ncols)
{
	_backend.init(nraw_folder.c_str(), ncols, false);

	if (_backend.load_index_from_disk() == false) {
		return false;
	}

	_real_nrows = _backend.get_number_rows();

	// _backend.print_indexes();

	return true;
}

//#define EXPORT_LINE_PARALLEL_REDUCE

#ifdef EXPORT_LINE_PARALLEL_REDUCE
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

	QString empty;
	QString line = tbb::parallel_deterministic_reduce(
		tbb::blocked_range<size_t>(tbb::blocked_range<size_t>(0, column_count, 1)),
		empty,
		[&](const tbb::blocked_range<size_t>& column, QString) -> QString {

			PVCol c = column.begin();
			size_t col = col_indexes[c];

			assert(idx < get_number_rows());
			QString v(at(idx, col));

			PVRush::PVUtils::safe_export(v, sep_char, quote_char);

			return v;
		},
		[&](const QString& left, const QString& right) -> QString {
			QString& l = const_cast<QString&>(left);
			l.append(sep_char);
			l.append(right);

			return l;
		}
	);

	return line;
}
#else
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
#endif

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

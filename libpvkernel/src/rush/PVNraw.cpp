/**
 * \file PVNraw.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVConfig.h>

#include <pvkernel/rush/PVNraw.h>
#include <pvkernel/rush/PVNrawException.h>
#include <pvkernel/rush/PVUtils.h>

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
const QString PVRush::PVNraw::default_tmp_path = QDir::tempPath() + "/picviz";
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
}

void PVRush::PVNraw::reserve(PVRow const nrows, PVCol const ncols)
{
	QSettings &pvconfig = PVCore::PVConfig::get().config();

	// Generate random path
	QString nraw_dir_base = pvconfig.value(config_nraw_tmp, default_tmp_path).toString() + QDir::separator() + nraw_tmp_pattern;
	QByteArray nstr = nraw_dir_base.toLocal8Bit();
	if (mkdtemp(nstr.data()) == nullptr) {
		throw PVNrawException(QObject::tr("unable to create temporary directory ") + nraw_dir_base);
	}

	_backend.init(nstr.constData(), ncols);

	if(nrows == 0) {
		_max_nrows = PICVIZ_LINES_MAX;
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

bool PVRush::PVNraw::add_chunk_utf16(PVCore::PVChunk const& chunk)
{
	if (_real_nrows == _max_nrows) {
		// the whole chunk can be skipped
		return false;
	}

	// Write all elements of the chunk in the final nraw
	PVCore::list_elts const& elts = chunk.c_elements();
	PVCore::list_elts::const_iterator it_elt;

	UErrorCode err = U_ZERO_ERROR;
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

		_real_nrows++;
		PVCol col = 0;
		PVCore::list_fields::const_iterator it_field;
		for (it_field = fields.begin(); it_field != fields.end(); it_field++) {
			// Convert to UTF8
			// TODO: make the whole process in utf8.. !
			PVCore::PVField const& field = *it_field;
			reserve_tmp_buf(field.size());
			const size_t size_utf8 = ucnv_fromUChars(_ucnv, _tmp_conv_buf, _tmp_conv_buf_size, (const UChar*) field.begin(), field.size()/sizeof(UChar), &err);
			if (!U_SUCCESS(err)) {
				PVLOG_WARN("Unable to convert field %d to UTF8! Field is ignored..\n", col);
				continue;
			}

			_backend.add(col, _tmp_conv_buf, size_utf8);
			col++;
		}
	}

	return true;
}

void PVRush::PVNraw::fit_to_content()
{
	_backend.flush();
	if (_real_nrows > PICVIZ_LINES_MAX) {
		_real_nrows = PICVIZ_LINES_MAX;
	}
}

QString PVRush::PVNraw::get_value(PVRow row, PVCol col, bool* complete /*= nullptr*/) const
{
	assert(row < get_number_rows());
	assert(col < get_number_cols());
	size_t size = 0;
	const char* buf = _backend.at_no_cache(row, col, size, complete);
	return QString::fromUtf8(buf, size);
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

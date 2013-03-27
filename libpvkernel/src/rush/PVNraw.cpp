/**
 * \file PVNraw.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVFileHelper.h>
#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/core/PVDirectory.h>

#include <pvkernel/rush/PVNraw.h>

#include <tbb/tick_count.h>

#include <iostream>

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#include <QDir>
#include <QDirIterator>
#include <QFileInfo>

#define DEFAULT_LINE_SIZE 100
#define MAX_SIZE_RESERVE (size_t(1024*1024*1024u)) // 1GB

#define CONFIG_NRAW_TMP "pvkernel/nraw_tmp"

#define NRAW_TMP_PATTERN "nraw-XXXXXX"
#define NRAW_TMP_NAME_REGEXP  "nraw-??????"

static const QString default_tmp_path = QDir::tempPath() + "/picviz";

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

void PVRush::PVNraw::reserve(PVRow const /*nrows*/, PVCol const ncols)
{
	// Generate random path
	QString nraw_dir_base = pvconfig.value(CONFIG_NRAW_TMP, default_tmp_path).toString() + QDir::separator() + NRAW_TMP_PATTERN;
	QByteArray nstr = nraw_dir_base.toLocal8Bit();
	mkdtemp(nstr.data());
	_backend.init(nstr.constData(), ncols);
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
		std::cout << qPrintable(nraw_line_to_csv(i)) << std::endl;
	}
}

void PVRush::PVNraw::dump_csv(const QString& file_path)
{
	FILE* file = fopen(qPrintable(file_path), "w");
	for (PVRow i = 0; i < get_number_rows(); i++) {
		const std::string& csv_line = nraw_line_to_csv(i).toStdString();
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

QString PVRush::PVNraw::get_value(PVRow row, PVCol col) const
{
	assert(row < get_number_rows());
	assert(col < get_number_cols());
	size_t size = 0;
	const char* buf = _backend.at(row, col, size);
	return QString::fromUtf8(buf, size);
}

QString PVRush::PVNraw::nraw_line_to_csv(PVRow idx) const
{
	assert(idx < get_number_rows());
	QString ret;
	static QLatin1String quote("\"");
	static QLatin1String escaped_quote("\\\"");
	static QLatin1String sep(",");
	PVCol c;
	for (c = 0; c < get_number_cols()-1; c++) {
		QString v(at(idx,c));
		bool do_quote = false;
		if (v.contains(QChar(','))) {
			do_quote = true;
		}
		if (v.contains(QChar('"'))) {
			do_quote = true;
			v.replace(quote, escaped_quote);
		}
		if (do_quote) {
			v.append(quote);
			v.prepend(quote);
		}
		ret += at(idx, c) + sep;
	}
	ret += at(idx, c);
	return ret;
}

QStringList PVRush::PVNraw::list_unused_nraw_directories(const QString &base_directory,
                                                         const QString &name_filter)
{
	QDir nraw_dir_base(base_directory, name_filter);
	QStringList result;

	for (const QString &sub_dir_name : nraw_dir_base.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
		QString sub_dir_path(base_directory + QDir::separator() + sub_dir_name);
		QDirIterator it(sub_dir_path,
		                QDir::Files | QDir::NoDotAndDotDot,
		                QDirIterator::Subdirectories);
		bool has_opened_file = false;

		while(it.hasNext()) {
			if (it.fileName().isEmpty()) {
				it.next();
				continue;
			}

			const char *c_file_name = it.filePath().toLocal8Bit().data();
			has_opened_file |= PVCore::PVFileHelper::is_already_opened(c_file_name);

			it.next();
		}

		if(has_opened_file == false) {
			result << base_directory + QDir::separator() + sub_dir_name;
		}
	}

	return result;
}

void PVRush::PVNraw::remove_unused_nraw_directories(const QString &base_directory,
                                                    const QString &name_filter)
{
	QStringList sl = list_unused_nraw_directories(base_directory, name_filter);

	for (const QString& dir : sl) {
		PVCore::PVDirectory::remove_rec(dir);
	}
}

void PVRush::PVNraw::remove_unused_nraw_directories()
{
	QString base_dir(pvconfig.value(CONFIG_NRAW_TMP,
	                                default_tmp_path).toString());
	QString regexp(NRAW_TMP_NAME_REGEXP);

	remove_unused_nraw_directories(base_dir, regexp);

}

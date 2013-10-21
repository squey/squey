/**
 * \file PVNraw.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVRUSH_NRAW_H
#define PVRUSH_NRAW_H

#include <QString>
#include <QStringList>
#include <QVector>

#include <vector>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/core/PVMatrix.h>
#include <pvkernel/core/PVMeanValue.h>
#include <pvkernel/core/PVUnicodeString.h>

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVNrawDiskBackend.h>

#include <tbb/tbb_allocator.h>
#include <tbb/tick_count.h>

extern "C" {
#include <unicode/ucsdet.h>
#include <unicode/ucnv.h>
}


namespace PVRush {

class LibKernelDecl PVNraw
{
public:
	static const QString config_nraw_tmp;
	static const QString default_tmp_path;
	static const QString nraw_tmp_pattern;
	static const QString nraw_tmp_name_regexp;

private:
	PVNraw& operator=(const PVNraw&) = delete;
	PVNraw(const PVNraw&) = delete;

public:
	// Unique values
	typedef PVNrawDiskBackend::unique_values_t unique_values_t;
	typedef PVNrawDiskBackend::unique_values_value_t unique_values_value_t;
	typedef PVNrawDiskBackend::unique_values_container_t unique_values_container_t;

	// Count by
	typedef PVNrawDiskBackend::count_by_t count_by_t;

public:
	PVNraw();
	~PVNraw();

	void reserve(PVRow const nrows, PVCol const ncols);
	void clear();

	// Move an nraw data to another PVNraw object. No copy and allocations occurs.
	inline PVRow get_number_rows() const { return _real_nrows; }
	inline PVCol get_number_cols() const { return _backend.get_number_cols(); }

	QString get_value(PVRow row, PVCol col) const;
	inline PVCore::PVUnicodeString at_unistr(PVRow row, PVCol col) const
	{
		assert(row < get_number_rows());
		assert(col < get_number_cols());
		size_t size;
		const char* buf = _backend.at(row, col, size);
		return PVCore::PVUnicodeString((PVCore::PVUnicodeString::utf_char*) buf, size);
	}

	inline PVCore::PVUnicodeString at_unistr_no_cache(PVRow row, PVCol col) const
	{
		assert(row < get_number_rows());
		assert(col < get_number_cols());
		size_t size;
		const char* buf = _backend.at_no_cache(row, col, size);
		return PVCore::PVUnicodeString((PVCore::PVUnicodeString::utf_char*) buf, size);
	}

	inline QString at(PVRow row, PVCol col) const { return get_value(row, col); }
	inline std::string at_string(PVRow row, PVCol col) const
	{
		assert(row < get_number_rows());
		assert(col < get_number_cols());
		size_t size;
		const char* buf = _backend.at(row, col, size);
		return std::string(buf, size);
	}

	bool add_chunk_utf16(PVCore::PVChunk const& chunk);

	template <class Iterator>
	bool add_column(Iterator /*begin*/, Iterator /*end*/)
	{
		return false;
	}

	inline QString get_axis_name(PVCol format_axis_id) const
	{
		if(format_axis_id < format->get_axes().size()) {
			return format->get_axes().at(format_axis_id).get_name();
		}
		return QString("");
	}

	void resize_nrows(PVRow const nrows)
	{
		if (nrows < _real_nrows) {
			_real_nrows = nrows;
		}
	}

	template <typename F>
	inline bool visit_column_tbb(PVCol const c, F const& f, tbb::task_group_context* ctxt = NULL) const
	{
		return _backend.visit_column_tbb(c, f, ctxt);
	}

	template <typename F>
	inline bool visit_column(PVCol const c, F const& f) const
	{
		return _backend.visit_column2(c, f);
	}

	template <typename F>
	inline bool visit_column_sel(PVCol const c, F const& f, PVCore::PVSelBitField const& sel) const
	{
		return _backend.visit_column2_sel(c, f, sel);
	}

	template <typename F>
	inline bool visit_column_tbb_sel(PVCol const c, F const& f, PVCore::PVSelBitField const& sel, tbb::task_group_context* ctxt = NULL) const
	{
		return _backend.visit_column_tbb_sel(c, f, sel, ctxt);
	}

	inline bool get_unique_values_for_col(PVCol const c, unique_values_t& ret, tbb::task_group_context* ctxt = NULL) const
	{
		return _backend.get_unique_values_for_col(c, ret, ctxt);
	}

	inline bool get_unique_values_for_col_with_sel(PVCol const c, unique_values_t& ret, PVCore::PVSelBitField const& sel, tbb::task_group_context* ctxt = NULL) const
	{
		return _backend.get_unique_values_for_col_with_sel(c, ret, sel, ctxt);
	}

	inline bool count_by_with_sel(PVCol const col1, PVCol const col2, count_by_t& ret, PVCore::PVSelBitField const& sel, size_t& v2_unique_values_count, tbb::task_group_context* ctxt = nullptr) const
	{
		return _backend.count_by_with_sel(col1, col2, ret, sel, v2_unique_values_count, ctxt);
	}

	QString nraw_line_to_csv(PVRow idx) const;
	QStringList nraw_line_to_qstringlist(PVRow idx) const;

	void fit_to_content();

	void dump_csv();
	void dump_csv(const QString& file_path);

	PVFormat_p& get_format() { return format; }
	PVFormat_p const& get_format() const { return format; }

	/**
	 * returns the folder path used for Nraw files
	 */
	const std::string& get_nraw_folder() const { return _backend.get_nraw_folder(); }

public:
	bool load_from_disk(const std::string& nraw_folder, PVCol ncols);

private:
	void clear_table();
	void reserve_tmp_buf(size_t n);

private:
	PVFormat_p format;
	PVRow _real_nrows;

	mutable PVNrawDiskBackend _backend;
	UConverter* _ucnv;

	char* _tmp_conv_buf;
	size_t _tmp_conv_buf_size;
};

}

#endif	/* PVRUSH_NRAW_H */

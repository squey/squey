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
	PVNraw& operator=(const PVNraw&) = delete;
	PVNraw(const PVNraw&) = delete;

public:
	PVNraw();
	~PVNraw();

	void reserve(PVRow const nrows, PVCol const ncols);
	void clear();

	// Move an nraw data to another PVNraw object. No copy and allocations occurs.
	static void swap(PVNraw &dst, PVNraw& src);

	inline PVRow get_number_rows() const { return _real_nrows; }
	inline PVCol get_number_cols() const { return _backend.get_number_cols(); }

	QString get_value(PVRow row, PVCol col) const;
	//void PVCore::PVUnicodeString const& at_unistr(PVRow row, PVCol col) const
	inline QString at(PVRow row, PVCol col) const { return get_value(row, col); }

	bool add_chunk_utf16(PVCore::PVChunk const& chunk);

	template <class Iterator>
	bool add_column(Iterator begin, Iterator end)
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

	QString nraw_line_to_csv(PVRow idx) const;
	QStringList nraw_line_to_qstringlist(PVRow idx) const;

	void fit_to_content();

	void dump_csv();

	PVFormat_p& get_format() { return format; }
	PVFormat_p const& get_format() const { return format; }

private:
	void clear_table();
	void reserve_tmp_buf(size_t n);

private:
	PVFormat_p format;
	PVRow _real_nrows;

	mutable PVNrawDiskBackend<> _backend;
	UConverter* _ucnv;

	char* _tmp_conv_buf;
	size_t _tmp_conv_buf_size;
};

}

#endif	/* PVRUSH_NRAW_H */

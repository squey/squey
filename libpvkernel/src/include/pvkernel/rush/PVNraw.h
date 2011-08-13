/*
 * $Id: PVNraw.h 3249 2011-07-05 11:14:53Z aguinet $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 */

#ifndef PVRUSH_NRAW_H
#define PVRUSH_NRAW_H

#include <QString>
#include <QStringList>
#include <QVector>

#include <vector>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVMeanValue.h>

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVNrawChild.h>

#include <tbb/scalable_allocator.h>

namespace PVRush {

	class LibKernelDecl PVNraw {
	public:
		typedef std::vector<QString, tbb::scalable_allocator<QString> > nraw_table_line;
		typedef std::vector<nraw_table_line, tbb::scalable_allocator<nraw_table_line> > nraw_table;
		typedef std::vector<nraw_table_line, tbb::scalable_allocator<nraw_table_line> > nraw_trans_table;
	private:
		QVector<PVNrawChild> children;
		PVCore::PVMeanValue<size_t> _mean_line_chars;
		PVRow _reserved_lines;

		// Buffer management
		QChar* _buf_strings;
		QChar* _cur_buf;
		size_t _rem_len_buf;
		size_t _len_buf;
		std::list<std::pair<QChar*, size_t> > _buf_todel;
	public:
		PVNraw();
		~PVNraw();

		void reserve(PVRow row);
		void create_trans_nraw();
		void free_trans_nraw();
		void clear();

		// This is explicit so that we are aware that we are going to allocate
		// a huge amount of memory !
		static void copy(PVNraw &dst, PVNraw const& src);

		// Move an nraw data to another PVNraw object. No copy and allocations occurs.
		static void move(PVNraw &dst, PVNraw& src);

		inline void push_line_chars(size_t nchars) { _mean_line_chars.push(nchars); }
		inline nraw_table& get_table() { return table; }
		inline nraw_table const& get_table() const { return table; }

		inline PVRow get_number_rows() const { return table.size(); }
		inline PVCol get_number_cols() const
		{
			if (table.size() == 0) {
				return 0;
			}

			return table[0].size();
		}

		nraw_table table;
		nraw_trans_table trans_table;

		PVFormat_p format;

		inline QString const& get_value(PVRow row, PVCol col) const
		{
			assert(row < table.size());
			return table.at(row)[col];
		}

		inline nraw_table_line& add_row(size_t nfields)
		{
			table.push_back(nraw_table_line());
			nraw_table_line& ret  = *(table.end()-1);
			ret.resize(nfields);
			return ret;
		}

		inline QString get_axis_name(PVCol format_axis_id) const
		{
			if(format_axis_id < format->get_axes().size()) {
                return format->get_axes().at(format_axis_id).get_name();
            }
            return QString("");
		}

		inline static void set_field(nraw_table_line& line, size_t index_field, const QChar* buf, size_t nchars)
		{
//			if (nchars > _rem_len_buf) {
//				// Need a reallocation
//				PVLOG_INFO("(PVRush::PVNraw) NRAW buffer is too small. Allocate a new one.\n");
//				
//				// Try to predicte the global buffer size
//				ssize_t nlines = _reserved_lines - table.size();
//				if (nlines <= 0) {
//					nlines = 10024; // Reserve for 10024 lines
//				}
//
//				size_t new_size = nlines * _mean_line_chars.compute_mean();
//				while  (new_size <= nchars) {
//					// Grow up by 5M of characters
//					new_size += 5*1024*1024;
//				}
//
//				allocate_buf(new_size);
//			}
//
//			QChar* copy = _cur_buf;
//			memcpy(copy, buf, nchars*sizeof(QChar));
//			_cur_buf += nchars + 1;
//			_rem_len_buf -= nchars;
			
			static tbb::scalable_allocator<QChar> alloc;
			QChar* copy = alloc.allocate(nchars);
			memcpy(copy, buf, nchars*sizeof(QChar));

			line[index_field].setRawData(copy, nchars);
		}

		QString nraw_line_to_csv(size_t idx) const;

	private:
		void allocate_buf(size_t nchars);
		void delete_buffers();
		void clear_table();

	private:
		PVNraw(const PVNraw& /*nraw*/) {}
		PVNraw& operator=(PVNraw const& /*nraw*/) {return *this;}

	};

}

#endif	/* PVRUSH_NRAW_H */

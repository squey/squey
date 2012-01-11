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
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/core/PVMatrix.h>
#include <pvkernel/core/PVMeanValue.h>
#include <pvkernel/core/PVUnicodeString.h>

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVNrawChild.h>

#include <tbb/tbb_allocator.h>
#include <tbb/tick_count.h>

namespace PVRush {

	class LibKernelDecl PVNraw {
	public:
//		typedef std::vector<QString, tbb::scalable_allocator<QString> > nraw_table_line;
//		typedef std::vector<nraw_table_line, tbb::scalable_allocator<nraw_table_line> > nraw_table;
//		typedef std::vector<nraw_table_line, tbb::scalable_allocator<nraw_table_line> > nraw_trans_table;
		typedef PVCore::PVMatrix<PVCore::PVUnicodeString, PVRow, PVCol, PVCore::PVMatrixAllocatorMmap > nraw_table;
		typedef nraw_table::line nraw_table_line;
		typedef nraw_table::const_line const_nraw_table_line;
		typedef nraw_table::transposed_type nraw_trans_table;

		typedef nraw_table::column trans_nraw_table_line;
		typedef nraw_table::const_column const_trans_nraw_table_line;
	private:
		typedef std::list<PVCore::PVChunk*, tbb::tbb_allocator<PVCore::PVChunk*> > list_chunks_t;
	public:
		PVNraw();
		~PVNraw();

		void reserve(PVRow row, PVCol col);
		bool create_trans_nraw();
		void free_trans_nraw();
		void clear();

		// Move an nraw data to another PVNraw object. No copy and allocations occurs.
		static void swap(PVNraw &dst, PVNraw& src);

		inline trans_nraw_table_line get_col(PVCol col) { return table.get_col(col); }
		inline const_trans_nraw_table_line get_col(PVCol col) const { return table.get_col(col); }

		inline nraw_table& get_table() { return table; }
		inline nraw_table const& get_table() const { return table; }

		inline nraw_trans_table& get_trans_table() { return trans_table; }
		inline nraw_trans_table const& get_trans_table() const { return trans_table; }

		inline PVRow get_number_rows() const { return table.get_nrows(); }
		inline PVCol get_number_cols() const { return table.get_ncols(); }

		PVFormat_p format;

		inline QString at(PVRow row, PVCol col) const { return get_value(row, col); }

		inline QString get_value(PVRow row, PVCol col) const
		{
			assert(row < table.get_nrows());
			assert(col < table.get_ncols());
			return table.at(row,col).get_qstr();
		}

		inline PVCore::PVUnicodeString const& at_unistr(PVRow row, PVCol col) const
		{
			assert(row < table.get_nrows());
			assert(col < table.get_ncols());
			return table.at(row,col);
		}

		inline void set_value(PVRow row, PVCol col, PVCore::PVUnicodeString const& str)
		{
			table.set_value(row, col, str);
		}

		inline bool add_row(PVCore::PVElement& elt, PVCore::PVChunk const& parent)
		{
			if (_real_nrows >= table.get_nrows()) {
				// Reallocation is necessary
				PVLOG_INFO("(PVNraw::add_row) reallocation of the NRAW table (element %d asked,  table size is %d).\n", _real_nrows, table.get_nrows());
				table.resize_nrows(_real_nrows + parent.c_elements().size(), PVCore::PVUnicodeString());
				PVLOG_INFO("(PVNraw::add_row) resizing done !\n");
				return true;
			}
			PVCore::list_fields& lf = elt.fields();
			if (table.get_ncols() < (PVCol) lf.size()) {
				PVLOG_WARN("(PVNraw::add_row) NRAW table has %d fields, and %d are requested.\n", table.get_ncols(), lf.size());
				if (_real_nrows == 0) {
					PVLOG_WARN("(PVNraw::add_row) that's the first element of the NRAW, resizing...\n");
					table.resize(table.get_nrows(), lf.size());
					PVLOG_DEBUG("(PVNraw::add_row) resizing done !\n");
				}
				else {
					PVLOG_WARN("(PVNraw::add_row) that's not the first element of the NRAW, this element is invalid ! Discard it...\n");
					return false;
				}
			}
			PVCore::PVUnicodeString* pfields = table.get_row_ptr(_real_nrows);
			PVCore::list_fields::iterator it;
			PVCol j = 0;
			for (it = lf.begin(); it != lf.end(); it++) {
				pfields[j].set_from_slice(*it);
				j++;
			}

			_real_nrows++;
			return true;
		}

		template <class Iterator>
		bool add_column(Iterator begin, Iterator end)
		{
			PVCol idx_new_col = get_number_cols();
			tbb::tick_count tstart = tbb::tick_count::now();
			if (!table.resize_ncols(get_number_cols() + 1)) {
				return false;
			}
			tbb::tick_count tend = tbb::tick_count::now();
			PVLOG_INFO("add_column: resize_ncols took %0.4fs.\n", (tend-tstart).seconds());

			Iterator it;
			PVRow i = 0;
			for (it = begin; it != end; it++) {
				table.set_value(i, idx_new_col, *it);
				i++;
			}

			return true;
		}

		inline QString get_axis_name(PVCol format_axis_id) const
		{
			if(format_axis_id < format->get_axes().size()) {
                return format->get_axes().at(format_axis_id).get_name();
            }
            return QString("");
		}

		QString nraw_line_to_csv(PVRow idx) const;

		void fit_to_content();

		bool resize_nrows(PVRow row)
		{
			return table.resize_nrows(row);
		}

		void dump_csv();

		inline void push_chunk_todelete(PVCore::PVChunk* chunk) { _chunks_todel->push_back(chunk); }

		// AG: should be protected w/ friends and everything...
		void take_realloc_buffers(PVCore::buf_list_t& list);

	private:
		void allocate_buf(size_t nchars);
		void delete_buffers();
		void clear_table();

	private:
		PVNraw(const PVNraw& /*nraw*/) {}
		PVNraw& operator=(PVNraw const& /*nraw*/) { return *this; }

	private:
		QVector<PVNrawChild> children;
		PVRow _real_nrows;
		list_chunks_t* _chunks_todel;

		nraw_table table;
		nraw_trans_table trans_table;

		// Reallocated buffers from PVElement objects
		PVCore::buf_list_t* _reallocated_buffers; // buf_list_t defined in PVBufferSlice.h
	};

}

#endif	/* PVRUSH_NRAW_H */

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

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVNrawChild.h>

#include <tbb/scalable_allocator.h>

namespace PVRush {

	class LibKernelDecl PVNraw {
	public:
//		typedef std::vector<QString, tbb::scalable_allocator<QString> > nraw_table_line;
//		typedef std::vector<nraw_table_line, tbb::scalable_allocator<nraw_table_line> > nraw_table;
//		typedef std::vector<nraw_table_line, tbb::scalable_allocator<nraw_table_line> > nraw_trans_table;
		typedef PVCore::PVMatrix<PVCore::PVField*, PVRow, PVCol> nraw_table;
		typedef nraw_table::line nraw_table_line;
		typedef nraw_table::const_line const_nraw_table_line;
		typedef nraw_table::transposed_type nraw_trans_table;
		typedef nraw_trans_table::line trans_nraw_table_line;
		typedef nraw_trans_table::const_line const_trans_nraw_table_line;
	private:
		typedef std::list<PVCore::PVChunk*, tbb::scalable_allocator<PVCore::PVChunk*> > list_chunks_t;
	public:
		PVNraw();
		~PVNraw();

		void reserve(PVRow row, PVCol col);
		bool create_trans_nraw();
		void free_trans_nraw();
		void clear();

		// Move an nraw data to another PVNraw object. No copy and allocations occurs.
		static void swap(PVNraw &dst, PVNraw& src);

		inline nraw_table& get_table() { return table; }
		inline nraw_table const& get_table() const { return table; }

		inline nraw_trans_table& get_trans_table() { return trans_table; }
		inline nraw_trans_table const& get_trans_table() const { return trans_table; }

		inline PVRow get_number_rows() const { return table.get_nrows(); }
		inline PVCol get_number_cols() const { return table.get_ncols(); }

		PVFormat_p format;

		inline QString const& at(PVRow row, PVCol col) const { return get_value(row, col); }

		inline QString const& get_value(PVRow row, PVCol col) const
		{
			assert(row < table.get_nrows());
			assert(col < table.get_ncols());
			return table.at(row,col)->get_qstr();
		}

		inline bool add_row(PVCore::PVElement& elt)
		{
			if (_real_nrows >= table.get_nrows()) {
				// Reallocation is necessary
				PVLOG_DEBUG("(PVNraw::add_row) reallocation of the NRAW table (element %d asked,  table size is %d).\n", _real_nrows, table.get_nrows());
				table.resize_nrows(_real_nrows + 60240);
				return true;
			}
			PVCore::list_fields& lf = elt.fields();
			if (table.get_ncols() < (PVCol) lf.size()) {
				PVLOG_WARN("(PVNraw::add_row) NRAW table has %d fields, and %d are requested.\n");
				if (_real_nrows == 0) {
					PVLOG_WARN("(PVNraw::add_row) that's the first element of the NRAW, resizing...\n");
					table.resize(table.get_nrows(), lf.size());
				}
				else {
					PVLOG_WARN("(PVNraw::add_row) that's not the first element of the NRAW, this element is invalid ! Discard it...\n");
					return false;
				}
			}
			PVCore::PVField** pfields = table.get_row_ptr(_real_nrows);
			PVCore::list_fields::iterator it;
			PVCol j = 0;
			for (it = lf.begin(); it != lf.end(); it++) {
				pfields[j] = &(*it);
				j++;
			}

			_real_nrows++;
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
	};

}

#endif	/* PVRUSH_NRAW_H */

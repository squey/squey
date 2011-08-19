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
#ifdef WIN32
		typedef std::vector<QString> nraw_table_line;
		typedef std::vector<nraw_table_line> nraw_table;
		typedef std::vector<nraw_table_line> nraw_trans_table;
#else
		typedef std::vector<QString, tbb::scalable_allocator<QString> > nraw_table_line;
		typedef std::vector<nraw_table_line, tbb::scalable_allocator<nraw_table_line> > nraw_table;
		typedef std::vector<nraw_table_line, tbb::scalable_allocator<nraw_table_line> > nraw_trans_table;
#endif
	public:
		PVNraw();
		~PVNraw();

		void reserve(PVRow row, PVCol col);
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
			_real_nrows++;
			return ret;
#if 0
			nraw_table_line* ret;
			if (_real_nrows >= table.size()) {
				// We always go a bit behind what's asked, so let's grow a bit our table
				PVRow prev_size = table.size();
				PVRow new_size = picviz_max(_real_nrows+1, prev_size+1000);
				table.resize(new_size);
				for (PVRow i = prev_size; i < new_size; i++) {
					table[i].resize(nfields);
				}
				ret = &table[_real_nrows];
			}
			else {
				ret = &table[_real_nrows];
				if (ret->size() == 0) {
					PVLOG_DEBUG("(PVNraw::add_row) the number of field was unknown. Resizing the nraw with %d fields...\n", nfields);
					// The number of field was unknown at the begging, so let's resize everything
					nraw_table::iterator it;
					for (it = table.begin(); it != table.end(); it++) {
						it->resize(nfields);
					}
				}
				else
				if (ret->size() != nfields) {
					PVLOG_ERROR("The number of fields asked (%d) is different from the one is the NRAW (%d) !! Resizing... (but clearly subefficient, and the NRAW won't be consistant (lines with a different number of fields)\n");
					ret->resize(nfields);
				}
			}
			_real_nrows++;
			return *ret;
#endif
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
			static tbb::scalable_allocator<QChar> alloc;
			QChar* copy = alloc.allocate(nchars);
			memcpy(copy, buf, nchars*sizeof(QChar));

			line[index_field].setRawData(copy, nchars);
		}

		QString nraw_line_to_csv(PVRow idx) const;

		void fit_to_content();

	private:
		void allocate_buf(size_t nchars);
		void delete_buffers();
		void clear_table();

	private:
		PVNraw(const PVNraw& /*nraw*/) {}
		PVNraw& operator=(PVNraw const& /*nraw*/) {return *this;}

	private:
		QVector<PVNrawChild> children;
		PVCore::PVMeanValue<size_t> _mean_line_chars;
		PVRow _real_nrows;
	};

}

#endif	/* PVRUSH_NRAW_H */

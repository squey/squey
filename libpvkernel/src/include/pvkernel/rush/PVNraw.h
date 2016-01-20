/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_NRAW_H
#define PVRUSH_NRAW_H

#include <QString>
#include <QStringList>
#include <QTextStream>

#include <vector>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVColumnIndexes.h>
#include <pvkernel/core/PVChunk.h>

#include <pvkernel/rush/PVFormat.h>

extern "C" {
#include <unicode/ucsdet.h>
#include <unicode/ucnv.h>
}

#include <pvcop/collection.h>
#include <pvcop/collector.h>
#include <pvcop/format.h>

namespace Inendi {
	class PVAxesCombination;
}

namespace PVCore {
	class PVSelBitField;
}

namespace PVRush {

class PVNraw
{
public:
	static const QString config_nraw_tmp;
	static const QString default_tmp_path;
	static const QString nraw_tmp_pattern;
	static const QString nraw_tmp_name_regexp;
	static const QString default_sep_char;
	static const QString default_quote_char;

private:
	PVNraw& operator=(const PVNraw&) = delete;
	PVNraw(const PVNraw&) = delete;

public:
	PVNraw();
	~PVNraw();

	void reserve(PVRow const nrows, PVCol const ncols);

	inline PVRow get_number_rows() const { return _real_nrows; }
	inline PVCol get_number_cols() const { return _format->column_count(); }

	/**
	 * Random access to an element in the NRaw.
	 */
	inline std::string at_string(PVRow row, PVCol col) const
	{
		assert(row < get_number_rows());
		assert(col < get_number_cols());
		return _collection->column(col).at(row);
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

	QStringList nraw_line_to_qstringlist(PVRow idx) const;

	void fit_to_content();


	QString export_line(
		PVRow idx,
		PVCore::PVColumnIndexes col_indexes = PVCore::PVColumnIndexes(),
		const QString sep_char = default_sep_char,
		const QString quote_char = default_quote_char
	) const;

	void export_lines(
		QTextStream& stream,
		const PVCore::PVSelBitField& sel,
		const PVCore::PVColumnIndexes& col_indexes,
		size_t start_index,
		size_t step_count,
		const QString sep_char = default_sep_char,
		const QString quote_char = default_quote_char
	) const;

	void dump_csv();
	void dump_csv(const QString& file_path);

	PVFormat_p& get_format() { return format; }
	PVFormat_p const& get_format() const { return format; }

	pvcop::collection& collection() { assert(_collection); return *_collection; }
	const pvcop::collection& collection() const { assert(_collection); return *_collection; }

public:
	bool load_from_disk(const std::string& nraw_folder, PVCol ncols);

private:
	void clear_table();
	const PVCore::PVColumnIndexes get_column_indexes() const;

private:
	PVFormat_p format;
	PVRow _real_nrows;
	PVRow _max_nrows;

	UConverter* _ucnv;

	std::unique_ptr<pvcop::collector> _collector = nullptr; //!< Structure to fill NRaw content.
	std::unique_ptr<pvcop::collection> _collection = nullptr; //!< Structure to read NRaw content.
	std::unique_ptr<pvcop::format> _format = nullptr; //!< Format with data management information.
};

}

#endif	/* PVRUSH_NRAW_H */

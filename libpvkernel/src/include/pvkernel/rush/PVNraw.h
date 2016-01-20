/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_NRAW_H
#define PVRUSH_NRAW_H

#include <fstream>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVColumnIndexes.h>

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
	class PVChunk;
}

namespace PVRush {

/**
 * Contains all informations to access imported data.
 *
 * Start in an invalid state as format is not known yet.
 * Then, we set format so we can create the collector (import struct)
 * Finally, data is imported, we don't need collector and use collection instead.
 */
class PVNraw
{
public:
	static const std::string config_nraw_tmp;
	static const std::string default_tmp_path;
	static const std::string nraw_tmp_pattern;
	static const std::string nraw_tmp_name_regexp;
	static const std::string default_sep_char;
	static const std::string default_quote_char;

private:
	/**
	 * Disable copy constructors.
	 */
	PVNraw& operator=(const PVNraw&) = delete;
	PVNraw(const PVNraw&) = delete;

public:
	PVNraw();
	~PVNraw();

	void reserve(PVRow const nrows, PVCol const ncols);

	/**
	 * Access layout of the NRaw.
	 *
	 * @warning: _real_nrows looks to be a copy from an invariang
	 * @fixme
	 */
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

	/**
	 * Insert data in the NRaw.
	 *
	 * @note: Input data is in utf16 and it is saved using utf8
	 * @note: We save a full chunk in a raw.
	 */
	bool add_chunk_utf16(PVCore::PVChunk const& chunk);

	/**
	 * Get the axis name from the format.
	 */
	inline std::string get_axis_name(PVCol format_axis_id) const
	{
		if(format_axis_id < format->get_axes().size()) {
			return format->get_axes().at(format_axis_id).get_name().toStdString();
		}
		return "";
	}

	/**
	 * Shrink the current NRaw to nrows values.
	 */
	void resize_nrows(PVRow const nrows)
	{
		if (nrows < _real_nrows) {
			_real_nrows = nrows;
		}
	}

	void fit_to_content();

	std::string export_line(
		PVRow idx,
		const PVCore::PVColumnIndexes& col_indexes,
		const std::string sep_char = default_sep_char,
		const std::string quote_char = default_quote_char
	) const;

	void export_lines(
		std::ofstream& stream,
		const PVCore::PVSelBitField& sel,
		const PVCore::PVColumnIndexes& col_indexes,
		size_t start_index,
		size_t step_count,
		const std::string& sep_char = default_sep_char,
		const std::string& quote_char = default_quote_char
	) const;

	void dump_csv(std::ostream &os=std::cout);
	void dump_csv(const std::string& file_path);

	/**
	 * Accessors
	 */
	PVFormat_p& get_format() { return format; }
	PVFormat_p const& get_format() const { return format; }

	pvcop::collection& collection() { assert(_collection); return *_collection; }
	const pvcop::collection& collection() const { assert(_collection); return *_collection; }

public:
	/**
	 * Create a NRaw from and NRaw folder on HDD.
	 */
	void load_from_disk(const std::string& nraw_folder);

private:
	PVFormat_p format; //!< Format with graphical information.
	PVRow _real_nrows; //!< Current number of line in the NRaw.
	PVRow _max_nrows;  //!< Maximum number of lines required.

	UConverter* _ucnv; //!< Converter from UTF16 to UTF8

	std::unique_ptr<pvcop::collector> _collector = nullptr; //!< Structure to fill NRaw content.
	std::unique_ptr<pvcop::collection> _collection = nullptr; //!< Structure to read NRaw content.
	std::unique_ptr<pvcop::format> _format = nullptr; //!< Format with data management information.
};

}

#endif	/* PVRUSH_NRAW_H */

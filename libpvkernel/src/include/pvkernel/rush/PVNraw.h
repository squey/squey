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
 *
 * We can say it has : invalide state, read state and write state.
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

	/**
	 * Access layout of the NRaw.
	 */
	inline PVRow get_number_rows() const { assert(_collection && "We should be in read state"); return _collection->row_count(); }
	inline PVCol get_number_cols() const { assert(_collection && "We should be in read state"); return _collection->column_count(); }

	/**
	 * Random access to an element in the NRaw.
	 */
	inline std::string at_string(PVRow row, PVCol col) const
	{
		assert(_collection && "We have to be in read state");
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
		if(format_axis_id >= format->get_axes().size()) {
			throw std::runtime_error("Invalid axis id, can't get its name.");
		}
		return format->get_axes().at(format_axis_id).get_name().toStdString();
	}

	/**
	 * Close the collector and start the collection as import is done.
	 */
	void load_done();

	/**
	 * Create collector and format to load content.
	 */
	void prepare_load(PVRow const nrows);

	/**
	 * Export asked line with a specific column ordering.
	 *
	 * Column ordering may differ from original ordering.
	 */
	std::string export_line(
		PVRow idx,
		const PVCore::PVColumnIndexes& col_indexes,
		const std::string sep_char = default_sep_char,
		const std::string quote_char = default_quote_char
	) const;

	/**
	 * Export step_count lines from start_index with a specific column ordering.
	 * Less lines may be output as it care about selection and not selected
	 * lines are not exported
	 *
	 * Column ordering may differ from original ordering.
	 */
	void export_lines(
		std::ofstream& stream,
		const PVCore::PVSelBitField& sel,
		const PVCore::PVColumnIndexes& col_indexes,
		size_t start_index,
		size_t step_count,
		const std::string& sep_char = default_sep_char,
		const std::string& quote_char = default_quote_char
	) const;

	/**
	 * Export the PVNraw with initial ordering.
	 */
	void dump_csv(std::ostream &os=std::cout);
	void dump_csv(const std::string& file_path);

	/**
	 * Accessors
	 */
	void set_format(PVFormat_p const& f) { format = f;}
	PVFormat_p const& get_format() const { return format; }

	pvcop::collection& collection()
	{
		assert(_collection && "we have to be in read state");
		return *_collection;
	}

	pvcop::collection const& collection() const
	{
		assert(_collection && "we have to be in read state");
		return *_collection;
	}

public:
	/**
	 * Create a NRaw from and NRaw folder on HDD.
	 */
	void load_from_disk(const std::string& nraw_folder);

private:
	/// Variable usefull for reading
	std::unique_ptr<pvcop::collection> _collection = nullptr; //!< Structure to read NRaw content.

	/// Variable usefull for loading
	PVRow _real_nrows; //!< Current number of line in the NRaw.
	PVRow _max_nrows;  //!< Maximum number of lines required.
	UConverter* _ucnv; //!< Converter from UTF16 to UTF8
	std::unique_ptr<pvcop::collector> _collector = nullptr; //!< Structure to fill NRaw content.
	std::unique_ptr<pvcop::format> _format = nullptr; //!< Format with data management information.

	/// Variable usefull for both
	PVFormat_p format; //!< Format with graphical information.
};

}

#endif	/* PVRUSH_NRAW_H */

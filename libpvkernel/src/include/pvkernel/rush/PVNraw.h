/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVRUSH_NRAW_H
#define PVRUSH_NRAW_H

#include <fstream>

#include <pvkernel/core/PVBinaryChunk.h>
#include <pvkernel/core/PVColumnIndexes.h>
#include <pvkernel/core/PVSelBitField.h>
#include <pvkernel/rush/PVControllerJob.h>

#include <pvcop/collection.h>
#include <pvcop/collector.h>

namespace pybind11
{
class array;
}

namespace PVCore
{
class PVSelBitField;
class PVTextChunk;
} // namespace PVCore

namespace PVRush
{

class NrawLoadingFail : public std::runtime_error
{
	using std::runtime_error::runtime_error;
};

class PVControllerJob;

/**
 * Contains all informations to access imported data.
 *
 * Start in an invalid state as format is not known yet.
 * Then, we set format so we can create the collector (import struct)
 * Finally, data is imported, we don't need collector and use collection instead.

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

  public:
	PVNraw();

	/**
	 * Disable copy constructor/assignment.
	 */
	PVNraw& operator=(const PVNraw&) = delete;
	PVNraw(const PVNraw&) = delete;

	/**
	 * But not move constructor/assignment.
	 */
	PVNraw(PVNraw&& other) = default;
	PVNraw& operator=(PVNraw&& other) = default;

	/**
	 * Access layout of the NRaw.
	 */
	inline PVRow row_count() const
	{
		if (_collection) {
			return _collection->row_count();
		} else if (_collector) {
			return _real_nrows;
		} else {
			return 0;
		}
	}
	inline PVCol column_count() const
	{
		assert(_collection && "We should be in read state");
		return PVCol(_collection->column_count());
	}

	/**
	 * Random access to an element in the NRaw.
	 */
	inline std::string at_string(PVRow row, PVCol col) const
	{
		assert(_collection && "We have to be in read state");
		assert(row < row_count());
		assert(col < column_count());
		return _columns[col].at(row);
	}

	const pvcop::db::array& column(PVCol col) const
	{
		assert(_collection && "we have to be in read state");
		assert(col < (PVCol)_columns.size());

		return _columns[col];
	}

	bool append_column(const pvcop::db::type_t& column_type, const pybind11::array& column);

	void delete_column(PVCol col);

	const pvcop::db::read_dict* column_dict(PVCol col) const { return _collection->dict(col); }

	std::string dir() const { return _collection->rootdir(); }

	/**
	 * Insert data in the NRaw.
	 *
	 * @note: Input data is in utf16 and it is saved using utf8
	 * @note: We save a full chunk in a raw.
	 */
	bool add_chunk_utf16(PVCore::PVTextChunk const& chunk);

	/**
	 * Insert data in the NRaw.
	 *
	 * @note: Input data is in binary using "pvcop::db::sink::column_chunk_t"
	 */
	bool add_bin_chunk(PVCore::PVBinaryChunk& chunk);

	/**
	 * Close the collector and start the collection as import is done.
	 * Also, compute a selection of valid elements.
	 */
	void load_done(const PVControllerJob::invalid_elements_t& inv_elts);

	/**
	 * Create collector and format to load content.
	 */
	void prepare_load(pvcop::formatter_desc_list const& format);

	/**
	 * Export asked line with a specific column ordering.
	 *
	 * Column ordering may differ from original ordering.
	 */
	std::string export_line(PVRow idx,
	                        const PVCore::PVColumnIndexes& col_indexes,
	                        const std::string sep_char = default_sep_char,
	                        const std::string quote_char = default_quote_char) const;

	/**
	 * Export the PVNraw with initial ordering.
	 */
	void dump_csv(const std::string& file_path = "") const;

	const PVCore::PVSelBitField& valid_rows_sel() const { return _valid_rows_sel; }
	size_t get_valid_row_count() const { return _valid_elements_count; }

  public:
	/**
	 * Create a NRaw from and NRaw folder on HDD.
	 */
	void load_from_disk(const std::string& nraw_folder);

	static PVNraw serialize_read(PVCore::PVSerializeObject& obj);
	void serialize_write(PVCore::PVSerializeObject& so) const;

  private:
	void init_collection(const std::string& path);

  private:
	/// Variable usefull for reading
	std::unique_ptr<pvcop::collection> _collection = nullptr; //!< Structure to read NRaw content.
	std::vector<pvcop::db::array> _columns;

	/// Variable usefull for loading
	size_t _real_nrows;                                     //!< Current number of line in the NRaw.
	std::unique_ptr<pvcop::collector> _collector = nullptr; //!< Structure to fill NRaw content.

	PVCore::PVSelBitField _valid_rows_sel;
	size_t _valid_elements_count;
};
} // namespace PVRush

#endif /* PVRUSH_NRAW_H */

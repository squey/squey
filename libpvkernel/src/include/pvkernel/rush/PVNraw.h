/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_NRAW_H
#define PVRUSH_NRAW_H

#include <fstream>

#include <pvkernel/core/PVColumnIndexes.h>
#include <pvkernel/core/PVSelBitField.h>
#include <pvkernel/rush/PVControllerJob.h>

#include <pvcop/collection.h>
#include <pvcop/collector.h>
#include <pvcop/types/exception/partially_converted_chunk_error.h>

namespace PVCore
{
class PVSelBitField;
class PVChunk;
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
	using unconvertable_values_t = pvcop::types::exception::partially_converted_chunk_error;

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
	 * Disable copy/move constructors.
	 */
	PVNraw& operator=(const PVNraw&) = delete;
	PVNraw(const PVNraw&) = delete;
	PVNraw(PVNraw&& other)
	    : _collection(std::move(other._collection))
	    , _unconvertable_values(std::move(other._unconvertable_values))
	    , _valid_rows_sel(std::move(other._valid_rows_sel))
	    , _valid_elements_count(other._valid_elements_count)
	{
		assert(not other._collector);
	}

	PVNraw& operator=(PVNraw&& other)
	{
		_collection = std::move(other._collection);
		_unconvertable_values = std::move(other._unconvertable_values);
		_valid_rows_sel = std::move(other._valid_rows_sel);
		_valid_elements_count = other._valid_elements_count;

		assert(not other._collector);
		assert(not _collector);
		return *this;
	}

	/**
	 * Access layout of the NRaw.
	 */
	inline PVRow get_row_count() const
	{
		if (_collection) {
			return _collection->row_count();
		} else if (_collector) {
			return _real_nrows;
		} else {
			return 0;
		}
	}
	inline PVCol get_number_cols() const
	{
		assert(_collection && "We should be in read state");
		return _collection->column_count();
	}

	/**
	 * Random access to an element in the NRaw.
	 */
	inline std::string at_string(PVRow row, PVCol col) const
	{
		assert(_collection && "We have to be in read state");
		assert(row < get_row_count());
		assert(col < get_number_cols());
		return _columns[col].at(row);
	}

	/**
	 * Insert data in the NRaw.
	 *
	 * @note: Input data is in utf16 and it is saved using utf8
	 * @note: We save a full chunk in a raw.
	 */
	bool add_chunk_utf16(PVCore::PVChunk const& chunk);

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
	void dump_csv(std::ostream& os = std::cout) const;
	void dump_csv(const std::string& file_path) const;

	/**
	 * Accessors
	 */
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

	const unconvertable_values_t& unconvertable_values() const { return _unconvertable_values; }

	void unconvertable_values_search(PVCol col,
	                                 const PVCore::PVSelBitField& in_sel,
	                                 PVCore::PVSelBitField& out_sel) const;

	void unconvertable_values_search(PVCol col,
	                                 const std::vector<std::string>& exps,
	                                 const PVCore::PVSelBitField& in_sel,
	                                 PVCore::PVSelBitField& out_sel) const;

	void unconvertable_values_search(
	    PVCol col,
	    const std::vector<std::string>& exps,
	    const PVCore::PVSelBitField& in_sel,
	    PVCore::PVSelBitField& out_sel,
	    std::function<bool(const std::string&, const std::string&)> predicate) const;

	void empty_values_search(PVCol col,
	                         const PVCore::PVSelBitField& in_sel,
	                         PVCore::PVSelBitField& out_sel) const;

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

	unconvertable_values_t _unconvertable_values;
	PVCore::PVSelBitField _valid_rows_sel;
	size_t _valid_elements_count;
};
} // namespace PVRush

#endif /* PVRUSH_NRAW_H */

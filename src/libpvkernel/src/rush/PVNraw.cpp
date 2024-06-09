//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVSelBitField.h>
#include <pvkernel/core/PVSerializeObject.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/core/PVTextChunk.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/rush/PVNraw.h>
#include <pvkernel/rush/PVNrawException.h>
#include <pvkernel/rush/PVUtils.h>
#include <pvcop/collector.h>
#include <pvcop/sink.h>
#include <pvcop/db/write_dict.h>
#include <pvcop/db/exceptions/invalid_collection.h>
#include <pvkernel/rush/PVCSVExporter.h>
#include <assert.h>
#include <qchar.h>
#include <qdir.h>
#include <qstring.h>
#include <stdint.h>
#include <stdlib.h>
#include <memory>
#include <numeric>
#include <algorithm>
#include <list>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "pvbase/types.h"
#include "pvcop/collection.h"
#include "pvcop/db/array.h"
#include "pvcop/db/sink.h"
#include "pvcop/db/types.h"
#include "pvkernel/core/PVBinaryChunk.h"
#include "pvkernel/core/PVColumnIndexes.h"
#include "pvkernel/rush/PVControllerJob.h"

namespace pvcop {
class formatter_desc_list;
}  // namespace pvcop
namespace pybind11 {
class array;
}  // namespace pybind11

const std::string PVRush::PVNraw::config_nraw_tmp = "pvkernel/nraw_tmp";
const std::string PVRush::PVNraw::default_tmp_path = "/tmp/squey";
const std::string PVRush::PVNraw::nraw_tmp_pattern = "nraw-XXXXXX";
const std::string PVRush::PVNraw::nraw_tmp_name_regexp = "nraw-??????";

/*****************************************************************************
 *
 * PVRush::PVNraw::PVNraw
 *
 ****************************************************************************/

PVRush::PVNraw::PVNraw() : _real_nrows(0), _valid_rows_sel(0) {}

/*****************************************************************************
 *
 * PVRush::PVNraw::prepare_load
 *
 ****************************************************************************/

void PVRush::PVNraw::prepare_load(pvcop::formatter_desc_list const& format)
{
	// Generate random path
	std::string collector_path =
	    PVRush::PVNrawCacheManager::nraw_dir().toStdString() + "/" + nraw_tmp_pattern;
	if (mkdtemp(&collector_path.front()) == nullptr) {
		throw PVNrawException("unable to create temporary directory " + collector_path);
	}

	// Create collector and format
	_collector = std::make_unique<pvcop::collector>(collector_path.data(), format);
	_collection.reset();
}

/*****************************************************************************
 *
 * PVRush::PVNraw::init_collection
 *
 ****************************************************************************/

void PVRush::PVNraw::init_collection(const std::string& path)
{
	_collection = std::make_unique<pvcop::collection>(path);

	/*
	 * map columns once and for all
	 */
	_columns.clear();
	for (size_t col = 0; col < _collection->column_count(); col++) {
		_columns.emplace_back(_collection->column(col));
	}
}


/*****************************************************************************
 *
 * PVRush::PVNraw::insert_column
 *
 ****************************************************************************/

bool PVRush::PVNraw::append_column(const pvcop::db::type_t& column_type, const pybind11::array& column)
{
	assert(_collection && "A collection must be open");
	bool ret = _collection->append_column(column_type, column);
	if (ret) {
		_columns.emplace_back(_collection->column(_collection->column_count()-1));
	}
	return ret;
}

/*****************************************************************************
 *
 * PVRush::PVNraw::delete_column
 *
 ****************************************************************************/

void PVRush::PVNraw::delete_column(PVCol col)
{
	_collection->delete_column(col);
}

/*****************************************************************************
 *
 * PVRush::PVNraw::add_chunk_utf16
 *
 ****************************************************************************/

bool PVRush::PVNraw::add_chunk_utf16(PVCore::PVTextChunk const& chunk)
{
	assert(_collector && "We have to be in read state");

	const size_t column_count = _collector->column_count();

	// Write all elements of the chunk in the final nraw
	PVCore::list_elts const& elts = chunk.c_elements();

	// Use the sink to write data from RAM to HDD
	pvcop::sink snk(*_collector);

	std::vector<pvcop::sink::field_t> pvcop_fields;
	pvcop_fields.reserve(elts.size() * column_count);

	// Count number of extracted line. It is not the same as the number of elements as some of them
	// may be invalid or empty or we may skip the end when enough data is extracted.
	PVRow local_row = elts.size();

	for (PVCore::PVElement* elt : elts) {

		PVCore::PVElement& e = *elt;
		assert(not e.filtered() and "We can't have filtered value in the Nraw");
		if (!e.valid()) {
			for (size_t i = 0; i < column_count; i++) {
				pvcop_fields.emplace_back(pvcop::sink::field_t());
			}
			continue;
		}

		PVCore::list_fields const& fields = e.c_fields();
		for (PVCore::PVField const& field : fields) {
			// Save the field
			pvcop_fields.emplace_back(pvcop::sink::field_t(field.begin(), field.size()));
		}
	}

	snk.write_chunk_by_row(chunk.agg_index(), local_row, pvcop_fields.data());

#pragma omp atomic
	_real_nrows += local_row;

	return true;
}

bool PVRush::PVNraw::add_bin_chunk(PVCore::PVBinaryChunk& chunk)
{
	{
	pvcop::db::sink snk(*_collector);
	snk.write_chunk_by_column(chunk.start_index(), chunk.rows_count(), chunk.columns_chunk());

	// Set string column dicts if any
	for (std::pair<PVCol, std::unique_ptr<pvcop::db::write_dict>>& col_dict : chunk.take_column_dicts()) {
		snk.set_column_dict(col_dict.first, std::move(col_dict.second));
	}
	if (chunk.optimized_invalid()) {
		for (size_t col = 0; col < chunk.columns_chunk().size(); ++col) {
			if (chunk.is_invalid(PVCol(col))) {
				const uint8_t* null_bitmap = chunk.null_bitmap(PVCol(col));
				snk.set_chunk_null_bitmap(PVCol(col), chunk.start_index(), chunk.rows_count(), null_bitmap);
			}
		}
	}
	}

	if (not chunk.optimized_invalid())
	{
		// Set null values if any
		pvcop::sink snk(*_collector);
		for (size_t col = 0; col < chunk.columns_chunk().size(); ++col) {
			if (chunk.is_invalid(PVCol(col))) {
				auto range =  chunk.invalids().equal_range(PVCol(col));
				for (auto it = range.first; it != range.second; ++it) {
					snk.set_invalid(col, chunk.start_index() + it->second);
				}
			}
		}
	}


#pragma omp atomic
	_real_nrows += chunk.rows_count();

	return true;
}

/*****************************************************************************
 *
 * PVRush::PVNraw::load_done
 *
 ****************************************************************************/
void PVRush::PVNraw::load_done(const PVControllerJob::invalid_elements_t& inv_elts)
{
	assert(_collector);

	// Close collector to be sure it is saved before we load it in the collection.
	_collector->close();

	// Compute selection of valid elements
	if (_real_nrows) {
		_valid_rows_sel = PVCore::PVSelBitField(_real_nrows);
		_valid_rows_sel.select_all();
		for (const auto& e : inv_elts) {
			_valid_rows_sel.set_line(e.first, false);
		}
		_valid_elements_count = _valid_rows_sel.bit_count();

		if (_valid_elements_count != 0) {
			// Create the collection only if there are imported lines.
			init_collection(_collector->rootdir());
		}
	}
	_collector.reset();
}

/*****************************************************************************
 *
 * PVRush::PVNraw::load_from_disk
 *
 ****************************************************************************/

void PVRush::PVNraw::load_from_disk(const std::string& nraw_folder)
{
	_collector.reset();

	/**
	 * to avoid leaking pvcop exception outside of PVNraw or PVRush, the
	 * collection opening failure is catch there and is reported according
	 * to the actual logic of error reporting in Squey stack.
	 *
	 * TODO: we will have to rething the error propagation from a library
	 * to an other in the factorization process. Rethrowing an exception
	 * PVRush::invalid_nraw could be a nicer solution than returning
	 * a boolean.
	 */
	try {
		init_collection(nraw_folder);
	} catch (pvcop::db::exception::invalid_collection&) {
		throw NrawLoadingFail("Can't creation a collection from disk");
	}

	_real_nrows = _collection->row_count();
}

/*****************************************************************************
 *
 * PVRush::PVNraw::dump_csv
 *
 ****************************************************************************/

void PVRush::PVNraw::dump_csv(std::string const& file_path /* = "" */) const
{
	PVCore::PVColumnIndexes cols(column_count());
	std::iota(cols.begin(), cols.end(), PVCol(0));
	PVCore::PVSelBitField sel(row_count());
	sel.select_all();

	PVRush::PVCSVExporter::export_func_f export_func =
	    [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
	        const std::string& quote) { return export_line(row, cols, sep, quote); };

	PVRush::PVCSVExporter exp(cols, row_count(), export_func);
	exp.export_rows(file_path, sel);
}

/*****************************************************************************
 *
 * PVRush::PVNraw::export_line
 *
 ****************************************************************************/

std::string PVRush::PVNraw::export_line(PVRow idx,
                                        const PVCore::PVColumnIndexes& col_indexes,
                                        const std::string sep_char /* = default_sep_char */,
                                        const std::string quote_char /* = default_quote_char */
                                        ) const
{
	assert(col_indexes.size() != 0);

	// Displayed column, not NRaw column
	std::string line;

	for (PVCol c : col_indexes) {
		line += PVRush::PVUtils::safe_export(at_string(idx, c), sep_char, quote_char) + sep_char;
	}

	// Remove last sep_char
	line.resize(line.size() - sep_char.size());

	return line;
}

void PVRush::PVNraw::serialize_write(PVCore::PVSerializeObject& so) const
{
	so.set_current_status("Saving raw data...");
	QString nraw_path = QString::fromStdString(_collection->rootdir());
	so.attribute_write("nraw_path", nraw_path);

	int vec = _valid_elements_count;
	so.attribute_write("valid_count", vec);

	so.set_current_status("Saving valid elements information...");
	PVCore::PVSerializeObject_p sel_obj = so.create_object("valid_elts");
	_valid_rows_sel.serialize_write(*sel_obj);
}

PVRush::PVNraw PVRush::PVNraw::serialize_read(PVCore::PVSerializeObject& so)
{
	so.set_current_status("Loading raw data...");
	PVRush::PVNraw nraw;
	auto nraw_folder = so.attribute_read<QString>("nraw_path");
	nraw_folder =
	    PVRush::PVNrawCacheManager::nraw_dir() + QDir::separator() + QDir(nraw_folder).dirName();
	nraw.load_from_disk(nraw_folder.toStdString());

	so.set_current_status("Loading invalid events information...");
	int vec = so.attribute_read<int>("valid_count");
	nraw._valid_elements_count = vec;
	PVCore::PVSerializeObject_p sel_obj = so.create_object("valid_elts");
	nraw._valid_rows_sel = PVCore::PVSelBitField::serialize_read(*sel_obj);

	return nraw;
}

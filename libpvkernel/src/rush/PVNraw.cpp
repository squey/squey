/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvbase/general.h>
#include <pvkernel/core/PVSelBitField.h>
#include <pvkernel/core/PVSerializeObject.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>
#include <pvkernel/core/PVChunk.h>
#include <pvkernel/core/PVExporter.h>

#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/rush/PVNraw.h>
#include <pvkernel/rush/PVNrawException.h>
#include <pvkernel/rush/PVUtils.h>

#include <pvcop/collector.h>
#include <pvcop/sink.h>

#include <pvcop/db/exceptions/invalid_collection.h>

#include <fstream>
#include <iterator>
#include <unordered_set>
#include <omp.h>

const std::string PVRush::PVNraw::config_nraw_tmp = "pvkernel/nraw_tmp";
const std::string PVRush::PVNraw::default_tmp_path = "/tmp/inendi";
const std::string PVRush::PVNraw::nraw_tmp_pattern = "nraw-XXXXXX";
const std::string PVRush::PVNraw::nraw_tmp_name_regexp = "nraw-??????";

/*****************************************************************************
 *
 * PVRush::PVNraw::PVNraw
 *
 ****************************************************************************/

PVRush::PVNraw::PVNraw() : _real_nrows(0), _valid_rows_sel(0)
{
}

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
	_collector.reset(new pvcop::collector(collector_path.data(), format));
	_collection.reset();
}

/*****************************************************************************
 *
 * PVRush::PVNraw::init_collection
 *
 ****************************************************************************/

void PVRush::PVNraw::init_collection(const std::string& path)
{
	_collection.reset(new pvcop::collection(path));

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
 * PVRush::PVNraw::add_chunk_utf16
 *
 ****************************************************************************/

bool PVRush::PVNraw::add_chunk_utf16(PVCore::PVChunk const& chunk)
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

	// TODO : We should check that EXTRACTED_ROW_COUNT_LIMIT is not reach.
	//        This is not trivial because chunks are written in parallel...

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

	try {
		snk.write_chunk_by_row(chunk.agg_index(), local_row, pvcop_fields.data());
	} catch (const pvcop::types::exception::partially_converted_chunk_error& e) {

#pragma omp critical
		_unconvertable_values.merge(e);
	}

#pragma omp atomic
	_real_nrows += local_row;

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
	 * to the actual logic of error reporting in the Inspector stack.
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

void PVRush::PVNraw::dump_csv(std::ostream& os) const
{
	PVCore::PVColumnIndexes cols(column_count());
	std::iota(cols.begin(), cols.end(), 0);
	PVCore::PVSelBitField sel(row_count());
	sel.select_all();

	PVCore::PVExporter::export_func export_func =
	    [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
	        const std::string& quote) { return export_line(row, cols, sep, quote); };

	PVCore::PVExporter exp(os, sel, cols, row_count(), export_func);
	exp.export_rows(0);
}

/*****************************************************************************
 *
 * PVRush::PVNraw::dump_csv
 *
 ****************************************************************************/

void PVRush::PVNraw::dump_csv(std::string const& file_path) const
{
	std::ofstream ofs(file_path);
	dump_csv(ofs);
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

	for (int c : col_indexes) {
		line += PVRush::PVUtils::safe_export(at_string(idx, c), sep_char, quote_char) + sep_char;
	}

	// Remove last sep_char
	line.resize(line.size() - sep_char.size());

	return line;
}

void PVRush::PVNraw::unconvertable_values_search(PVCol col,
                                                 const PVCore::PVSelBitField& in_sel,
                                                 PVCore::PVSelBitField& out_sel) const
{
	auto const& bad_values = _unconvertable_values.bad_conversions();

	for (auto const& bad_value : bad_values) {

		PVRow row = bad_value.first;
		if (in_sel.get_line(row)) {

			auto const& bad_cols = bad_value.second;
			if (bad_cols.find(col) != bad_cols.end()) {
				out_sel.set_bit_fast(row);
			}
		}
	}
}

void PVRush::PVNraw::unconvertable_values_search(PVCol col,
                                                 const std::vector<std::string>& exps,
                                                 const PVCore::PVSelBitField& in_sel,
                                                 PVCore::PVSelBitField& out_sel) const
{
	std::unordered_set<std::string> e(exps.begin(), exps.end());

	auto const& bad_values = _unconvertable_values.bad_conversions();

	for (auto const& bad_value : bad_values) {

		PVRow row = bad_value.first;
		if (in_sel.get_line(row)) {

			auto const& bad_cols = bad_value.second;
			auto it = bad_cols.find(col);
			if (it != bad_cols.end() && e.find(it->second) != e.end()) {
				out_sel.set_bit_fast(row);
			}
		}
	}
}

void PVRush::PVNraw::unconvertable_values_search(
    PVCol col,
    const std::vector<std::string>& exps,
    const PVCore::PVSelBitField& in_sel,
    PVCore::PVSelBitField& out_sel,
    std::function<bool(const std::string&, const std::string&)> predicate) const
{
	auto const& bad_values = _unconvertable_values.bad_conversions();

	for (auto const& bad_value : bad_values) {

		PVRow row = bad_value.first;
		if (in_sel.get_line(row)) {

			auto const& bad_cols = bad_value.second;
			auto it = bad_cols.find(col);

			if (it != bad_cols.end()) {
				const std::string& invalid_value = it->second;
				bool match = std::any_of(exps.begin(), exps.end(), [&](const std::string& exp) {
					return predicate(invalid_value, exp);
				});
				if (match) {
					out_sel.set_bit_fast(row);
				}
			}
		}
	}
}

void PVRush::PVNraw::empty_values_search(PVCol col,
                                         const PVCore::PVSelBitField& in_sel,
                                         PVCore::PVSelBitField& out_sel) const
{
	auto const& empty_values = _unconvertable_values.empty_conversions();

	for (auto const& empty_value : empty_values) {
		PVRow row = empty_value.first;
		if (in_sel.get_line(row)) {
			auto const& empty_cols = empty_value.second;
			if (empty_cols.find(col) != empty_cols.end()) {
				out_sel.set_bit_fast(row);
			}
		}
	}
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

	// Serialize invalid value
	so.set_current_status("Saving uncorrectly converted elements information...");
	int idx = 0;
	auto const& bad_values = _unconvertable_values.bad_conversions();
	int bad_conv_row_count = bad_values.size();
	so.attribute_write("bad_conv/row_count", bad_conv_row_count);
	for (auto const& bad_value : bad_values) {
		so.attribute_write("bad_conv/" + QString::number(idx) + "/row", bad_value.first);
		auto const& bad_cols = bad_value.second;
		int bad_conv_col_count = bad_cols.size();
		so.attribute_write("bad_conv/" + QString::number(idx) + "/col_count", bad_conv_col_count);
		int idx_col = 0;
		for (auto const& bad_col : bad_cols) {
			so.attribute_write("bad_conv/" + QString::number(idx) + "/" + QString::number(idx_col) +
			                       "/col",
			                   bad_col.first);
			QString value = QString::fromStdString(bad_col.second);
			so.attribute_write("bad_conv/" + QString::number(idx) + "/" + QString::number(idx_col) +
			                       "/value",
			                   value);
			idx_col++;
		}
		idx++;
	}

	so.set_current_status("Saving empty fields information...");
	auto const& empty_values = _unconvertable_values.empty_conversions();

	std::vector<PVCore::PVSelBitField> sels(column_count(), PVCore::PVSelBitField(row_count()));
	for (size_t col = 0; col < sels.size(); col++) {
		sels[col].select_none();
	}

	std::unordered_set<size_t> empty_cols_indexes;

	for (auto const& empty_value : empty_values) {
		PVRow row = empty_value.first;
		auto const& empty_cols = empty_value.second;
		for (auto const& empty_col : empty_cols) {
			sels[empty_col].set_bit_fast(row);
			empty_cols_indexes.insert(empty_col);
		}
	}

	std::stringstream str_col_indexes;
	std::copy(empty_cols_indexes.begin(), empty_cols_indexes.end(),
	          std::ostream_iterator<size_t>(str_col_indexes, ","));

	so.attribute_write("empty_conv/columns", QString::fromStdString(str_col_indexes.str()));

	for (PVCol col : empty_cols_indexes) {
		PVCore::PVSerializeObject_p sel_obj =
		    so.create_object("empty_conv_col_" + QString::number(col));
		sels[col].serialize_write(*sel_obj);
	}
}

PVRush::PVNraw PVRush::PVNraw::serialize_read(PVCore::PVSerializeObject& so)
{
	so.set_current_status("Loading raw data...");
	PVRush::PVNraw nraw;
	QString nraw_folder = so.attribute_read<QString>("nraw_path");
	nraw_folder =
	    PVRush::PVNrawCacheManager::nraw_dir() + QDir::separator() + QDir(nraw_folder).dirName();
	nraw.load_from_disk(nraw_folder.toStdString());

	so.set_current_status("Loading invalid events information...");
	int vec = so.attribute_read<int>("valid_count");
	nraw._valid_elements_count = vec;
	PVCore::PVSerializeObject_p sel_obj = so.create_object("valid_elts");
	nraw._valid_rows_sel = PVCore::PVSelBitField::serialize_read(*sel_obj);

	// Serialize invalid values
	so.set_current_status("Loading uncorrectly converted elements information...");
	int bad_conv_row_count = so.attribute_read<int>("bad_conv/row_count");
	for (int i = 0; i < bad_conv_row_count; i++) {
		int row = so.attribute_read<int>("bad_conv/" + QString::number(i) + "/row");
		int bad_conv_col_count =
		    so.attribute_read<int>("bad_conv/" + QString::number(i) + "/col_count");
		for (int j = 0; j < bad_conv_col_count; j++) {
			int col = so.attribute_read<int>("bad_conv/" + QString::number(i) + "/" +
			                                 QString::number(j) + "/col");
			QString value = so.attribute_read<QString>("bad_conv/" + QString::number(i) + "/" +
			                                           QString::number(j) + "/value");
			nraw._unconvertable_values.add(row, col, value.toStdString());
		}
	}

	so.set_current_status("Loading empty elements information...");
	QString str_col_indexes = so.attribute_read<QString>("empty_conv/columns");
	QStringList list_col_indexes = str_col_indexes.split(",", QString::SkipEmptyParts);
	std::vector<size_t> empty_cols_indexes;
	for (const QString& str_idx : list_col_indexes) {
		empty_cols_indexes.emplace_back(str_idx.toUInt());
	}

	for (size_t col = 0; col < empty_cols_indexes.size(); col++) {

		PVCore::PVSerializeObject_p sel_obj =
		    so.create_object("empty_conv_col_" + QString::number(empty_cols_indexes[col]));
		PVCore::PVSelBitField sel = PVCore::PVSelBitField::serialize_read(*sel_obj);

		for (size_t row = 0; row < sel.count(); row++) {
			if (sel.get_line_fast(row)) {
				nraw._unconvertable_values.add(row, empty_cols_indexes[col], "");
			}
		}
	}

	return nraw;
}

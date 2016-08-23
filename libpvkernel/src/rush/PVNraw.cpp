/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvbase/general.h>
#include <pvkernel/core/PVSelBitField.h>
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
			_collection.reset(new pvcop::collection(_collector->rootdir()));
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
		_collection.reset(new pvcop::collection(nraw_folder));
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
	PVCore::PVColumnIndexes cols(get_number_cols());
	std::iota(cols.begin(), cols.end(), 0);
	PVCore::PVSelBitField sel(get_row_count());
	sel.select_all();

	PVCore::PVExporter::export_func export_func =
	    [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
	        const std::string& quote) { return export_line(row, cols, sep, quote); };

	PVCore::PVExporter exp(os, sel, cols, get_row_count(), export_func);
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

void PVRush::PVNraw::serialize_write(PVCore::PVSerializeObject& so)
{
	QString nraw_path = QString::fromStdString(collection().rootdir());
	so.attribute("nraw_path", nraw_path);

	int vec = _valid_elements_count;
	so.attribute("valid_count", vec);

	PVCore::PVSerializeObject_p sel_obj = so.create_object("valid_elts", "valid_elts", true, true);
	_valid_rows_sel.serialize_write(*sel_obj);

	// Serialiaze invalide value
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

	idx = 0;
	auto const& empty_values = _unconvertable_values.empty_conversions();
	int empty_conv_row_count = empty_values.size();
	so.attribute_write("empty_conv/row_count", empty_conv_row_count);
	for (auto const& empty_value : empty_values) {
		so.attribute_write("empty_conv/" + QString::number(idx) + "/row", empty_value.first);
		auto const& empty_cols = empty_value.second;
		int empty_conv_col_count = empty_cols.size();
		so.attribute_write("empty_conv/" + QString::number(idx) + "/col_count",
		                   empty_conv_col_count);
		int idx_col = 0;
		for (auto const& empty_col : empty_cols) {
			so.attribute_write("empty_conv/" + QString::number(idx) + "/" +
			                       QString::number(idx_col++) + "/col",
			                   empty_col);
		}
		idx++;
	}
}

PVRush::PVNraw PVRush::PVNraw::serialize_read(PVCore::PVSerializeObject& so)
{
	PVRush::PVNraw nraw;
	QString nraw_folder;
	so.attribute("nraw_path", nraw_folder, QString());
	nraw_folder =
	    PVRush::PVNrawCacheManager::nraw_dir() + QDir::separator() + QDir(nraw_folder).dirName();
	nraw.load_from_disk(nraw_folder.toStdString());

	int vec;
	so.attribute("valid_count", vec);
	nraw._valid_elements_count = vec;
	PVCore::PVSerializeObject_p sel_obj = so.create_object("valid_elts", "valid_elts", true, true);
	nraw._valid_rows_sel = PVCore::PVSelBitField::serialize_read(*sel_obj);

	// Serialiaze invalide value
	int bad_conv_row_count;
	so.attribute("bad_conv/row_count", bad_conv_row_count);
	for (int i = 0; i < bad_conv_row_count; i++) {
		int row;
		so.attribute("bad_conv/" + QString::number(i) + "/row", row);
		int bad_conv_col_count;
		so.attribute("bad_conv/" + QString::number(i) + "/col_count", bad_conv_col_count);
		for (int j = 0; j < bad_conv_col_count; j++) {
			int col;
			so.attribute("bad_conv/" + QString::number(i) + "/" + QString::number(j) + "/col", col);
			QString value;
			so.attribute("bad_conv/" + QString::number(i) + "/" + QString::number(j) + "/value",
			             value);
			nraw._unconvertable_values.add(row, col, value.toStdString());
		}
	}

	int empty_conv_row_count;
	so.attribute("empty_conv/row_count", empty_conv_row_count);
	for (int i = 0; i < empty_conv_row_count; i++) {
		int row;
		so.attribute("empty_conv/" + QString::number(i) + "/row", row);
		int empty_conv_col_count;
		so.attribute("empty_conv/" + QString::number(i) + "/col_count", empty_conv_col_count);
		for (int j = 0; j < empty_conv_col_count; j++) {
			int col;
			so.attribute("empty_conv/" + QString::number(i) + "/" + QString::number(j) + "/col",
			             col);
			nraw._unconvertable_values.add(row, col, "");
		}
	}
	return nraw;
}

/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVSelBitField.h>
#include <pvkernel/core/PVElement.h>
#include <pvkernel/core/PVField.h>

#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/rush/PVNraw.h>
#include <pvkernel/rush/PVNrawException.h>
#include <pvkernel/rush/PVUtils.h>

#include <pvcop/collector.h>
#include <pvcop/sink.h>

#include <pvcop/db/exceptions/invalid_collection.h>

#include <iostream>
#include <fstream>
#include <omp.h>

const std::string PVRush::PVNraw::config_nraw_tmp = "pvkernel/nraw_tmp";
const std::string PVRush::PVNraw::default_tmp_path = "/tmp/inendi";
const std::string PVRush::PVNraw::nraw_tmp_pattern = "nraw-XXXXXX";
const std::string PVRush::PVNraw::nraw_tmp_name_regexp = "nraw-??????";
const std::string PVRush::PVNraw::default_sep_char = ",";
const std::string PVRush::PVNraw::default_quote_char = "\"";

/*****************************************************************************
 *
 * PVRush::PVNraw::PVNraw
 *
 ****************************************************************************/

PVRush::PVNraw::PVNraw():
	_real_nrows(0),
	_invalid_count(0)
{
	UErrorCode status = U_ZERO_ERROR;
	_ucnv = ucnv_open("UTF8", &status);
}

PVRush::PVNraw::~PVNraw()
{
	ucnv_close(_ucnv);
}

/*****************************************************************************
 *
 * PVRush::PVNraw::prepare_load
 *
 ****************************************************************************/

void PVRush::PVNraw::prepare_load(PVRow const nrows)
{
	// Generate random path
	std::string collector_path = PVRush::PVNrawCacheManager::nraw_dir().toStdString() + "/" + nraw_tmp_pattern;
	if (mkdtemp(&collector_path.front()) == nullptr) {
		throw PVNrawException("unable to create temporary directory " + collector_path);
	}

	// Create collector and format
	_collector.reset(new pvcop::collector(collector_path.data(), get_format()->get_storage_format()));
	_collection.reset();

	// Define maximum number of row;
	if(nrows == 0) {
		_max_nrows = INENDI_LINES_MAX;
	} else {
		_max_nrows = nrows;
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

	if (chunk.agg_index() > _max_nrows) {
		// the whole chunk can be skipped as we extracted enough data.
		return false;
	}

	const size_t column_count = _collector->column_count();

	// Write all elements of the chunk in the final nraw
	PVCore::list_elts const& elts = chunk.c_elements();

	// Use the sink to write data from RAM to HDD
	pvcop::sink snk(*_collector);

	std::vector<pvcop::sink::field_t> pvcop_fields;
	pvcop_fields.reserve(elts.size() * column_count);

	size_t remaining_fields_count = _max_nrows - _real_nrows;

	// Count number of extracted line. It is not the same as the number of elements as some of them
	// may be invalid or empty or we may skip the end when enough data is extracted.
	PVRow local_row = elts.size();
	for (PVCore::PVElement* elt: elts) {

		PVCore::PVElement& e = *elt;
		if (!e.valid()) {
			for(int i=0; i<column_count; i++) {
				std::unique_ptr<char[]> tmp_buf(new char[1]);
				tmp_buf[0] = '\0';
				pvcop_fields.emplace_back(pvcop::sink::field_t{std::move(tmp_buf), 1});
			}
			continue;
		}

		PVCore::list_fields const& fields = e.c_fields();
		if (fields.size() == 0) {
			for(int i=0; i<column_count; i++) {
				std::unique_ptr<char[]> tmp_buf(new char[1]);
				tmp_buf[0] = '\0';
				pvcop_fields.emplace_back(pvcop::sink::field_t{std::move(tmp_buf), 1});
			}
			continue;
		}
// FIXME : No cancel anymore in the middle of a chunk.
//		if (local_row == remaining_fields_count) {
//			/* we have enough events, skips the others. As the
//			 * chunk has been partially saved, the current chunked
//			 * index has to be saved by the caller (PVNrawOutput).
//			 */
//			break;
//		}

		assert(column_count == fields.size());
		for (PVCore::PVField const& field :fields) {
			// TODO: make the whole process in utf8.. !
			// Convert field to UT8
			std::unique_ptr<char[]> tmp_buf(new char[field.size()]);
			UErrorCode err = U_ZERO_ERROR;
			size_t size_utf8 = ucnv_fromUChars(_ucnv, tmp_buf.get(), field.size(), (const UChar*) field.begin(), field.size()/sizeof(UChar), &err);
			if (!U_SUCCESS(err)) {
				PVLOG_WARN("Unable to convert a field to UTF8! Field is ignored..\n");
				continue;
			}

			// Save the field
			pvcop_fields.emplace_back(pvcop::sink::field_t{std::move(tmp_buf), size_utf8});
		}
//		local_row++;
	}


	int i = snk.write_chunk_by_row(chunk.agg_index(), local_row, pvcop_fields.data());

#pragma omp atomic
	_invalid_count += i;
#pragma omp atomic
	_real_nrows += local_row;

	return true;
}

/*****************************************************************************
 *
 * PVRush::PVNraw::load_done
 *
 ****************************************************************************/
void PVRush::PVNraw::load_done()
{
	assert(_collector);
	assert(_real_nrows <= INENDI_LINES_MAX);

	// Close collector to be sure it is saved before we load it in the collection.
	_collector->close();

	if(_real_nrows != 0) {
		// Create the collection only if there are imported lines.
		_collection.reset(new pvcop::collection(_collector->rootdir()));
	}
	_collector.reset();
}

/*****************************************************************************
 *
 * PVRush::PVNraw::load_from_disk
 *
 ****************************************************************************/

bool PVRush::PVNraw::load_from_disk(const std::string& nraw_folder)
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
		return false;
	}

	_real_nrows = _collection->row_count();

	return true;
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
	PVCore::PVSelBitField sel;
	sel.select_all();
	export_lines(os, sel, cols, 0, get_row_count());
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
	static std::string escaped_quote("\\" + quote_char);

	assert(col_indexes.size() != 0);

	// Displayed column, not NRaw column
	std::string line;

	for(int c: col_indexes) {
		line += PVRush::PVUtils::safe_export(at_string(idx, c), sep_char, quote_char) + sep_char;
	}

	// Remove last sep_char
	line.resize(line.size() - sep_char.size());

	return line;
}

/*****************************************************************************
 *
 * PVRush::PVNraw::export_lines
 *
 ****************************************************************************/

void PVRush::PVNraw::export_lines(
	std::ostream& stream,
	const PVCore::PVSelBitField& sel,
	const PVCore::PVColumnIndexes& col_indexes,
	size_t start_index,
	size_t step_count,
	const std::string& sep_char /* = default_sep_char */,
	const std::string& quote_char /* = default_quote_char */
) const
{
	assert(get_number_cols() > 0);
	assert(col_indexes.size() != 0);
	// volatile as it will be modify by another thread.
	int volatile current_thread = 0;

	// Parallelize export algo:
	// Each thread have a local string. Thanks to static scheduling, first thread
	// will handle N first line, second one, N to 2N, ...
	// Finally, these string will be written in stream in thread order.
#pragma omp parallel
	{
		std::string content;
#pragma omp for schedule(static) nowait
		for (PVRow line_index = start_index; line_index < start_index + step_count; line_index++) {

			if (!sel.get_line_fast(line_index)) {
				continue;
			}

			content += export_line(line_index, col_indexes, sep_char, quote_char) + "\n";
		}

		// Data is in content but we lock here to make sure it is written ordered.
		// Ordered reduction may be available in OpenMP 4.0
		while(omp_get_thread_num() != current_thread);

		stream << content;
		current_thread++; // The next thread can do it.
	}

	std::flush(stream); // explicitely flush the stream
}

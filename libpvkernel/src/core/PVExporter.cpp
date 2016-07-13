/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#include <pvkernel/core/PVExporter.h>

#include <pvkernel/core/PVSelBitField.h>

#include <omp.h>

const std::string PVCore::PVExporter::default_sep_char = ",";
const std::string PVCore::PVExporter::default_quote_char = "\"";

PVCore::PVExporter::PVExporter(std::ostream& os,
                               const PVCore::PVSelBitField& sel,
                               const PVCore::PVColumnIndexes& column_indexes,
                               PVRow step_count,
                               export_func f,
                               const std::string& sep_char /* = default_sep_char */,
                               const std::string& quote_char /* = default_quote_char */
                               )
    : _os(os)
    , _sel(sel)
    , _column_indexes(column_indexes)
    , _step_count(step_count)
    , _sep_char(sep_char)
    , _quote_char(quote_char)
    , _f(f)
{
	assert(_column_indexes.size() != 0);
}

void PVCore::PVExporter::export_rows(size_t start_index)
{
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
		for (PVRow row_index = start_index; row_index < start_index + _step_count; row_index++) {

			if (!_sel.get_line_fast(row_index)) {
				continue;
			}

			content += _f(row_index, _column_indexes, _sep_char, _quote_char) + "\n";
		}

		// Data is in content but we lock here to make sure it is written ordered.
		// Ordered reduction may be available in OpenMP 4.0
		while (omp_get_thread_num() != current_thread)
			;

		_os << content;
		current_thread++; // The next thread can do it.
	}

	std::flush(_os); // explicitely flush the stream
}

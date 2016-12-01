/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef __EXPORT_SELECTION_H__
#define __EXPORT_SELECTION_H__

#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVControllerJob.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVUtils.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/core/PVExporter.h>
#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/core/PVStreamingCompressor.h>

#include <inendi/PVSelection.h>
#include <inendi/PVScene.h>
#include <inendi/PVMapped.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVView.h>

#include <cstdlib>
#include <iostream>

#include <QFile>

#include "common.h"

static constexpr const PVRow STEP_COUNT = 1000;

std::string
import_export(const std::string& input_file, const std::string& format, bool cancel = false)
{
	std::string file_extension = input_file.substr(input_file.rfind('.') + 1);

	pvtest::TestEnv env(input_file, format, 1, pvtest::ProcessUntil::View);

	bool delete_nraw_parent_dir = false;
	QDir nraw_dir(QString::fromStdString(PVRush::PVNraw::default_tmp_path));
	if (!nraw_dir.exists()) {
		nraw_dir.mkdir(QString::fromStdString(PVRush::PVNraw::default_tmp_path));
		delete_nraw_parent_dir = true;
	}

	Inendi::PVView* view = env.root.current_view();

	// Export selection to temporary file
	Inendi::PVSelection sel(view->get_row_count());
	sel.select_all();

	char temp_pattern[] = "/tmp/fileXXXXXX";
	close(mkstemp(temp_pattern));
	std::remove(temp_pattern);
	std::string output_file = std::string(temp_pattern) + "." + file_extension;

	PVRush::PVNraw& nraw = view->get_rushnraw_parent();
	const PVCore::PVColumnIndexes& col_indexes =
	    view->get_parent<Inendi::PVSource>().get_format().get_axes_comb();

	PVCore::PVExporter::export_func export_func =
	    [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
	        const std::string& quote) { return nraw.export_line(row, cols, sep, quote); };

	auto start = std::chrono::system_clock::now();

	PVCore::PVExporter exp(output_file, sel, col_indexes, nraw.row_count(), export_func);

	PVRow starting_row = 0;
	const PVRow nrows = nraw.row_count();
	PVRow step_count = std::min(STEP_COUNT, nrows);

	while (true) {
		starting_row = sel.find_next_set_bit(starting_row, nrows);
		if (starting_row == PVROW_INVALID_VALUE) {
			break;
		}

		step_count = std::min(step_count, nrows - starting_row);
		exp.set_step_count(step_count);
		exp.export_rows(starting_row);
		starting_row += step_count;

		if (cancel) {
			exp.cancel();
			break;
		}
	}
	exp.wait_finished();

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	std::cout << diff.count() << std::flush;

	// Remove nraw folder
	PVCore::PVDirectory::remove_rec(delete_nraw_parent_dir ? nraw_dir.path()
	                                                       : QString::fromStdString(nraw.dir()));
	return output_file;
}

#endif // __EXPORT_SELECTION_H__

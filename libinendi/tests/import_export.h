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
#include <pvkernel/rush/PVCSVExporter.h>
#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/core/PVStreamingCompressor.h>

#include <inendi/PVSelection.h>
#include <inendi/PVScene.h>
#include <inendi/PVMapped.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVView.h>
#include <inendi/PVPythonInterpreter.h>

#include <cstdlib>
#include <iostream>

#include <QFile>
#include <QTimer>

#include "common.h"

#include <filesystem>

static constexpr const PVRow STEP_COUNT = 1000;

std::string
import_export(const std::string& input_file, const std::string& format, bool test_selection, bool cancel = false)
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

	// Execute Python script if any
	bool is_path, disabled;
	Inendi::PVSource& src = view->get_parent<Inendi::PVSource>();
	QString python_script = src.get_original_format().get_python_script(is_path, disabled);
	if (is_path) {
		python_script.insert(0, (std::filesystem::current_path().string() + "/").c_str());
	}
	if (not disabled and not python_script.isEmpty()) {
		if (is_path and not QFileInfo(python_script).exists()) {
			assert(false && "Missing Python script");
		}
		else {
			Inendi::PVPythonInterpreter python_interpreter(src.get_parent<Inendi::PVRoot>());
			python_interpreter.execute_script(python_script.toStdString(), is_path);
		}
	}

	// Export selection to temporary file
	Inendi::PVSelection sel(view->get_row_count());
	if (test_selection) {
		sel.select_odd();
	}
	else {
		sel.select_all();
	}
	view->set_selection_view(sel);

	char temp_pattern[] = "/tmp/fileXXXXXX";
	close(mkstemp(temp_pattern));
	std::remove(temp_pattern);
	std::string output_file = std::string(temp_pattern) + "." + file_extension;

	PVRush::PVNraw& nraw = view->get_rushnraw_parent();
	const PVCore::PVColumnIndexes& col_indexes =
	    view->get_parent<Inendi::PVSource>().get_format().get_axes_comb();

	PVRush::PVCSVExporter::export_func_f export_func =
	    [&](PVRow row, const PVCore::PVColumnIndexes& cols, const std::string& sep,
	        const std::string& quote) { return nraw.export_line(row, cols, sep, quote); };
	PVRush::PVCSVExporter exp(col_indexes, nraw.row_count(), export_func);

	auto start = std::chrono::system_clock::now();

	if (cancel) {
		exp.cancel();
	}
	exp.export_rows(output_file, sel);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;

	std::cout << diff.count() << std::flush;

	// Remove nraw folder
	PVCore::PVDirectory::remove_rec(delete_nraw_parent_dir ? nraw_dir.path()
	                                                       : QString::fromStdString(nraw.dir()));
	return output_file;
}

#endif // __EXPORT_SELECTION_H__

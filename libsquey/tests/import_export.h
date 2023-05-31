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
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/core/PVStreamingCompressor.h>

#include <squey/PVSelection.h>
#include <squey/PVScene.h>
#include <squey/PVMapped.h>
#include <squey/PVPlotted.h>
#include <squey/PVView.h>
#include <squey/PVPythonInterpreter.h>

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

	Squey::PVView* view = env.root.current_view();

	// Execute Python script if any
	bool is_path, disabled;
	Squey::PVSource& src = view->get_parent<Squey::PVSource>();
	QString python_script = src.get_original_format().get_python_script(is_path, disabled);
	if (is_path) {
		python_script.insert(0, (std::filesystem::current_path().string() + "/").c_str());
	}
	if (not disabled and not python_script.isEmpty()) {
		if (is_path and not QFileInfo(python_script).exists()) {
			assert(false && "Missing Python script");
		}
		else {
			Squey::PVPythonInterpreter& python_interpreter = Squey::PVPythonInterpreter::get(src.get_parent<Squey::PVRoot>());
			python_interpreter.execute_script(python_script.toStdString(), is_path);
		}
	}

	// Export selection to temporary file
	Squey::PVSelection sel(view->get_row_count());
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
	    view->get_parent<Squey::PVSource>().get_format().get_axes_comb();

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

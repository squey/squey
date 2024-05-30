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

#ifndef __RUSH_LOGTSTATS_H__
#define __RUSH_LOGTSTATS_H__

#include <iomanip>
#include <vector>
#include <string>
#include <numeric>

#include "../../tests/rush/common.h"

#include <pvcop/db/algo.h>

#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/rush/PVUtils.h>

#include <QDomDocument>
#include <QFile>
#include <QTextStream>

struct integers_list {
	using integers_t = std::vector<int32_t>;
	integers_t values;
};

struct cmd_options {
	std::vector<std::string> log_files;
	std::string format_file;
	integers_list columns_index;
	bool extended_stats;
	size_t max_stats_rows;
};

/**
 * Remove unselected axes from format in order to speedup the import process
 */
std::string format_with_ignored_axes(const std::string& format_path, integers_list& cols)
{
	std::string tmp_format = pvtest::get_tmp_filename();

	QDomDocument dom;

	QFile format_file(QString::fromStdString(format_path));
	dom.setContent(&format_file);

	QDomNodeList axes_nodes = dom.elementsByTagName("axis");
	size_t original_axes_count = axes_nodes.count();

	// Select all columns if no colums were specified
	if (cols.values.empty()) {
		cols.values.resize(original_axes_count);
		std::iota(cols.values.begin(), cols.values.end(), 0);
	}

	if (not axes_nodes.isEmpty()) {
		for (int i = original_axes_count - 1; i >= 0; i--) {
			if (std::find(cols.values.begin(), cols.values.end(), i) == cols.values.end()) {
				QDomNode node = axes_nodes.at(i);
				node.parentNode().removeChild(node);
			}
		}
	}

	QFile tmp_format_file(QString::fromStdString(tmp_format));
	tmp_format_file.open(QIODevice::WriteOnly | QIODevice::Text);
	QTextStream tmp_format_stream(&tmp_format_file);
	tmp_format_stream << dom.toString();

	return tmp_format;
}

void run_stats(cmd_options opts)
{
	std::string tmp_collector_path;
	std::string format = format_with_ignored_axes(opts.format_file, opts.columns_index);

	{
		pvtest::TestEnv env(opts.log_files, format);
		env.load_data();

		const PVRush::PVFormat format = env._format;
		const PVRush::PVNraw& nraw = env._nraw;
		tmp_collector_path = nraw.dir();
		size_t row_count = env._nraw.valid_rows_sel().bit_count();

		pvlogger::info() << row_count << " rows / " << nraw.row_count() << std::endl;

		pvcop::db::array distinct_values;
		pvcop::db::array distinct_values_count;

		for (PVCol col(0); col < nraw.column_count(); col++) {
			const pvcop::db::array& column = nraw.column(col);
			pvcop::db::algo::distinct(column, distinct_values, distinct_values_count,
			                          env._nraw.valid_rows_sel());

			std::stringstream stream;
			stream << format.get_axes()[col].get_name().toStdString() << " : "
			       << distinct_values.size();

			if (distinct_values.size() > 1) {
				std::cout << '\t' << stream.str() << std::endl;

				if (opts.extended_stats) {
					pvcop::db::indexes indexes = distinct_values_count.parallel_sort();
					const auto& sorted = indexes.to_core_array();

					const auto& counts = distinct_values_count.to_core_array<uint64_t>();

					for (size_t row = 0; row < std::min(opts.max_stats_rows, indexes.size());
					     row++) {
						size_t sorted_row = sorted[sorted.size() - row - 1];
						std::cout << PVRush::PVUtils::safe_export(distinct_values.at(sorted_row),
						                                          ",", "\"")
						          << "," << counts[sorted_row] << "," << std::setprecision(1)
						          << std::fixed << (double)counts[sorted_row] / row_count * 100
						          << "%" << std::endl;
					}
				}
			} else {
				std::cout << stream.str() << " (" << distinct_values.at(0) << ")" << std::endl;
			}
		}
	}

	// Cleanup temporary nraw directory and format
	PVCore::PVDirectory::remove_rec(tmp_collector_path.c_str());
	QFile(QString::fromStdString(format)).remove();
}

#endif // __RUSH_LOGTSTATS_H__

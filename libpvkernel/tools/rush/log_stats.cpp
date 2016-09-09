/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include "../../tests/rush/common.h"

#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/rush/PVUtils.h>

#include <pvcop/db/algo.h>

#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>

#include <iomanip>
#include <sstream>

struct integers_list {
	using integers_t = std::vector<int32_t>;
	integers_t values;
};

void validate(boost::any& v, std::vector<std::string> const& options, integers_list*, int32_t)
{
	if (v.empty()) {
		v = boost::any(integers_list());
	}

	integers_list* p = boost::any_cast<integers_list>(&v);

	static constexpr const char* separator = ",";
	boost::char_separator<char> sep(separator);

	for (std::string const& option : options) {
		if (option.find(separator)) {
			boost::tokenizer<boost::char_separator<char>> tok(option, sep);
			for (const std::string& str : tok) {
				p->values.emplace_back(std::atoi(str.c_str()));
			}
		} else {
			p->values.emplace_back(std::atoi(option.c_str()));
		}
	}
}

namespace po = boost::program_options;

struct cmd_options {
	std::string log_file;
	std::string format_file;
	integers_list columns_index;
	bool extended_stats;
	size_t max_stats_rows;
};

cmd_options parse_options(int argc, char** argv)
{
	cmd_options opts;

	po::options_description desc("Get basic statistics from log files.\n\n"
	                             "Options");
	desc.add_options()("log_file", po::value<std::string>(&opts.log_file)->required(), "log file")(
	    "format_file", po::value<std::string>(&opts.format_file)->required(), "format file")(
	    "columns,c", po::value<integers_list>(&opts.columns_index), "column indexes")(
	    "extended-stats,x", po::bool_switch(&opts.extended_stats)->default_value(false),
	    "extended statistics")("max-stat-rows,m",
	                           po::value<size_t>(&opts.max_stats_rows)->default_value(10),
	                           "maximum distinct values to display for extended statistics")(
	    "help,h", "Show help message");

	po::positional_options_description pos_opts;
	pos_opts.add("log_file", 1);
	pos_opts.add("format_file", 1);

	po::variables_map vm;
	bool cmdline_error = false;

	try {
		po::store(po::command_line_parser(argc, argv)
		              .options(desc)
		              .positional(pos_opts)
		              .style(po::command_line_style::unix_style)
		              .run(),
		          vm);
		po::notify(vm);
	} catch (boost::program_options::required_option& e) {
		cmdline_error = true;
	} catch (boost::program_options::error& e) {
		cmdline_error = true;
	}

	if (vm.count("help") || cmdline_error) {
		std::cout << "Usage: " << basename(argv[0]) << " <log_file> <format_file> "
		                                               "[--columns=0,1,2,3,...] [--extended-stats] "
		                                               "[--max-stat-rows=10] \n";
		std::cout << desc;

		exit(-1);
	}

	return opts;
}

void run_stats(cmd_options opts)
{
	std::string tmp_collector_path;

	{
		pvtest::TestEnv env(opts.log_file, opts.format_file);
		env.load_data();

		const PVRush::PVFormat format = env._format;
		const pvcop::collection& collection = env._nraw.collection();
		tmp_collector_path = collection.rootdir();
		size_t row_count = pvcop::core::algo::bit_count(env._nraw.valid_rows_sel());

		pvlogger::info() << opts.log_file << " : " << row_count << " rows / "
		                 << collection.row_count() << std::endl;

		pvcop::db::array distinct_values;
		pvcop::db::array distinct_values_count;

		if (opts.columns_index.values.empty()) {
			opts.columns_index.values.resize(collection.column_count());
			std::iota(opts.columns_index.values.begin(), opts.columns_index.values.end(), 0);
		}

		for (size_t col : opts.columns_index.values) {
			const pvcop::db::array& column = collection.column(col);
			pvcop::db::algo::distinct(column, distinct_values, distinct_values_count,
			                          env._nraw.valid_rows_sel());

			std::stringstream stream;
			stream << format.get_axes()[col].get_name().toStdString() << " : "
			       << distinct_values.size();

			if (distinct_values.size() > 1) {
				std::cout << '\t' << stream.str() << std::endl;

				if (opts.extended_stats) {
					pvcop::db::indexes indexes =
					    pvcop::db::indexes::parallel_sort(distinct_values_count);
					const auto& sorted = indexes.to_core_array();

					const auto& counts = distinct_values_count.to_core_array<uint64_t>();

					for (size_t row = 0; row < std::min(opts.max_stats_rows, indexes.size());
					     row++) {
						size_t sorted_row = sorted[sorted.size() - row - 1];
						std::cout << PVRush::PVUtils::safe_export(distinct_values.at(sorted_row),
						                                          ",", "\"")
						          << "," << counts[sorted_row] << "," << std::setprecision(2)
						          << std::fixed << (double)counts[sorted_row] / row_count * 100
						          << "%" << std::endl;
					}
				}
			} else {
				std::cout << stream.str() << " (" << distinct_values.at(0) << ")" << std::endl;
			}
		}
	}

	// Cleanup temporary nraw directory
	PVCore::PVDirectory::remove_rec(tmp_collector_path.c_str());
}

int main(int argc, char** argv)
{
	run_stats(parse_options(argc, argv));
}

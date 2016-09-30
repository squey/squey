/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include "log_stats.h"

#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>

#include <iomanip>
#include <sstream>

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

cmd_options parse_options(int argc, char** argv)
{
	cmd_options opts;

	po::options_description desc("Get basic statistics from log files.\n\n"
	                             "Options");
	desc.add_options()("format_file", po::value<std::string>(&opts.format_file)->required(),
	                   "format file")(
	    "log_files", po::value<std::vector<std::string>>(&opts.log_files)->multitoken()->required(),
	    "log files")("columns,c", po::value<integers_list>(&opts.columns_index), "column indexes")(
	    "extended-stats,x", po::bool_switch(&opts.extended_stats)->default_value(false),
	    "extended statistics")("max-stat-rows,m",
	                           po::value<size_t>(&opts.max_stats_rows)->default_value(10),
	                           "maximum distinct values to display for extended statistics")(
	    "help,h", "Show help message");

	po::positional_options_description pos_opts;
	pos_opts.add("format_file", 1);
	pos_opts.add("log_files", 2);

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
		std::cout << "Usage: " << basename(argv[0]) << " <format_file> <log_files...> "
		                                               "[--columns=0,1,2,3,...] [--extended-stats] "
		                                               "[--max-stat-rows=10] \n";
		std::cout << desc;

		exit(-1);
	}

	return opts;
}

int main(int argc, char** argv)
{

/*
 * Override environment variables when deploying the tool to bypass the fact
 * that we are in fact using the internal test API.
 * The correct fix is probably to use the import API in Inendi namespace...
 */
#ifndef INENDI_DEVELOPER_MODE
	setenv("INENDI_PLUGIN_PATH", PLUGINS_DISTRIB_DIR, 1);
	setenv("PVKERNEL_PLUGIN_PATH", PLUGINS_DISTRIB_DIR, 1);
#endif

	run_stats(parse_options(argc, argv));
}

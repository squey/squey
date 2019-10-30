/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/rush/PVUtils.h>

#include "common.h"

static constexpr const char* csv_win1 = TEST_FOLDER "/sources/windows_endings1.csv";
static constexpr const char* csv_win2 = TEST_FOLDER "/sources/windows_endings2.csv";
static constexpr const char* csv_win3 = TEST_FOLDER "/sources/windows_endings3.csv";
static constexpr const char* csv_win_format = TEST_FOLDER "/formats/windows_endings.format";
static constexpr const char* exported_win_csv_file =
    TEST_FOLDER "/exports/multi_inputs_windows_endings.csv";

static constexpr const char* csv_empty_last_col1 = TEST_FOLDER "/sources/empty_last_column1.csv";
static constexpr const char* csv_empty_last_col2 = TEST_FOLDER "/sources/empty_last_column2.csv";
static constexpr const char* csv_empty_last_col_format =
    TEST_FOLDER "/formats/empty_last_column.format";
static constexpr const char* exported_empty_last_col_csv_file =
    TEST_FOLDER "/exports/multi_inputs_empty_last_column.csv";

int main()
{
	pvtest::TestEnv env(std::vector<std::string>{csv_win1, csv_win2, csv_win3}, csv_win_format);
	env.add_source(std::vector<std::string>{csv_empty_last_col1, csv_empty_last_col2},
	               csv_empty_last_col_format);

	const auto& sources = env.root.get_children<Inendi::PVSource>();
	PV_VALID(sources.size(), 2UL);

	auto exports =
	    std::vector<std::string>{exported_win_csv_file, exported_empty_last_col_csv_file};
	for (size_t i = 0; i < exports.size(); i++) {
		std::string out_path = pvtest::get_tmp_filename();
		// Dump the NRAW to file and check value is the same
		auto source_it = sources.begin();
		std::advance(source_it, i);
		const PVRush::PVNraw& nraw = (*source_it)->get_rushnraw();
		nraw.dump_csv(out_path);

		pvlogger::info() << exports[i] << " - " << out_path << std::endl;
		PV_ASSERT_VALID(PVRush::PVUtils::files_have_same_content(exports[i], out_path));

		std::remove(out_path.c_str());
	}

	return 0;
}

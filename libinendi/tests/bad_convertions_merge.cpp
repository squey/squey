/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2016
 */

#include <inendi/PVSource.h>

#include <pvkernel/core/inendi_assert.h>

#include "common.h"

constexpr static size_t ROW_COUNT = 100000;

static constexpr const char* log_file = TEST_FOLDER "/pvkernel/rush/bad_convertions_merge.csv";
static constexpr const char* log_format =
    TEST_FOLDER "/pvkernel/rush/bad_convertions_merge.csv.format";

int main()
{
	pvtest::TestEnv env(log_file, log_format);

	Inendi::PVSource* source = env.compute_mapping().get_parent<Inendi::PVSource>();

	for (size_t i = 0; i < ROW_COUNT; i++) {
		if (i < ROW_COUNT / 2) {
			PV_VALID(source->get_input_value(i, 0), std::string("test"));
		} else {
			PV_VALID(source->get_input_value(i, 0), std::string("0.0.0.0"));
		}
		PV_VALID(source->get_input_value(i, 1), std::string(""));
	};

	return 0;
}

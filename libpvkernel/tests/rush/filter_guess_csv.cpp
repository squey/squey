
#include <pvkernel/core/inendi_assert.h>

#include "test-env.h"
#include "common_guess.h"

int main(int argc, char** argv)
{
	if (argc != 5) {
		std::cerr << "Usage: " << argv[0] << "fields_count sep quote file" << std::endl;
		return 1;
	}

	init_env();

	PVCol found_fields_count(0);
	PVCol expected_fields_count = (PVCol)atoi(argv[1]);
	QString expected_sep(argv[2]);
	QString expected_quote(argv[3]);

	auto filter = pvtest::guess_filter(argv[4], found_fields_count);

	// if expected_fields_count == 0, the filter must be null
	if (expected_fields_count == 0) {
		PV_ASSERT_VALID(filter.get() == nullptr);

		return 0;
	}

	/* at that point, the filter must non-null, being just sure it is
	 */
	PV_ASSERT_VALID(filter.get() != nullptr);

	/* checking the found splitter is the CSV one
	 */
	PV_ASSERT_VALID(filter->registered_name() == "csv");

	/* checking the guessed fields count
	 */
	PV_VALID(found_fields_count, expected_fields_count);

	/* checking guessed CSV splitter parameters (seq and quote)
	 */
	const PVCore::PVArgumentList& args = filter->get_args();

	QString found_sep = args.at("sep").toString();
	QString found_quote = args.at("quote").toString();

	PV_ASSERT_VALID(found_sep == expected_sep, "found_sep", found_sep.toStdString(), "expected_sep",
	                expected_sep.toStdString());
	PV_ASSERT_VALID(found_quote == expected_quote, "found_quote", found_quote.toStdString(),
	                "expected_quote", expected_quote.toStdString());

	return 0;
}

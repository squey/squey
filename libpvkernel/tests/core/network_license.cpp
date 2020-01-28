/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2020
 */

#include <License.h>

#include <pvbase/general.h>
#include <pvkernel/core/inendi_assert.h>

int main()
{
	static constexpr const char license_server[] = "27000@licenses.srv.picviz";

	try {
		Inendi::Utils::License::RAII_InitLicense license_manager(license_server);
		Inendi::Utils::License::RAII_LicenseFeature full_program_license(INENDI_LICENSE_PREFIX,
		                                                                 INENDI_LICENSE_FEATURE);
	} catch (const Inendi::Utils::License::NotAvailableFeatureException& e) {
		PV_ASSERT_VALID(false);
	}

	PV_ASSERT_VALID(true);

	try {
		Inendi::Utils::License::RAII_InitLicense license_manager("12345@bad_server");
		Inendi::Utils::License::RAII_LicenseFeature full_program_license(INENDI_LICENSE_PREFIX,
		                                                                 INENDI_LICENSE_FEATURE);
	} catch (const Inendi::Utils::License::NotAvailableFeatureException& e) {
		PV_VALID((int)e.status_code, (int)Inendi::Utils::License::NotAvailableFeatureException::
		                                 STATUS_CODE::UNABLE_TO_RESOLVE_HOSTNAME);

		return 0;
	}

	PV_ASSERT_VALID(false);

	try {
		Inendi::Utils::License::RAII_InitLicense license_manager("27001@licenses.srv.picviz");
		Inendi::Utils::License::RAII_LicenseFeature full_program_license(INENDI_LICENSE_PREFIX,
		                                                                 INENDI_LICENSE_FEATURE);
	} catch (const Inendi::Utils::License::NotAvailableFeatureException& e) {
		PV_VALID((int)e.status_code, (int)Inendi::Utils::License::NotAvailableFeatureException::
		                                 STATUS_CODE::UNABLE_TO_CONTACT_SERVER);

		return 0;
	}

	PV_ASSERT_VALID(false);
}

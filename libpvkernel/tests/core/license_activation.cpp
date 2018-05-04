/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2018
 */
#include <pvkernel/core/PVLicenseActivator.h>
#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/core/PVDirectory.h>

#include <pvbase/general.h>

#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>

static constexpr const char online_license_path[] = "/tmp/online_inspector/online_inspector.lic";
static constexpr const char offline_license_path[] = "/tmp/offline_inspector/offline_inspector.lic";

QString online_license_folder = QFileInfo(QString::fromStdString(online_license_path)).dir().path();
QString offline_license_folder =
    QFileInfo(QString::fromStdString(offline_license_path)).dir().path();

std::string locking_code = PVCore::PVLicenseActivator::get_locking_code();

void test_online_trial_activation()
{
	PVCore::PVLicenseActivator::EError ret_code;

	// Activation service unavailable (NO_INTERNET_CONNECTION)
	char* https_proxy = std::getenv("https_proxy");
	setenv("https_proxy", "https://0.0.0.0", true); // Simulate internet disconnection
	ret_code = PVCore::PVLicenseActivator(online_license_path)
	               .online_activation("validated@testsuite.com", "0000-*000 0000 0000 0000",
	                                  "demo_pcap-inspector");
	setenv("https_proxy", https_proxy ? https_proxy : "", true); // reset HTTPS Proxy value
	PV_ASSERT_VALID(ret_code == PVCore::PVLicenseActivator::EError::NO_INTERNET_CONNECTION);

	// Activation service unavailable (ACTIVATION_SERVICE_UNAVAILABLE)
	ret_code = PVCore::PVLicenseActivator(online_license_path)
	               .online_activation("validated-generate@testsuite.com",
	                                  "0000-*000 0000 0000 0000", "demo_pcap-inspector_dry-run");
	PV_ASSERT_VALID(ret_code == PVCore::PVLicenseActivator::EError::ACTIVATION_SERVICE_UNAVAILABLE);

	// Unregistered mail (UNKOWN_USER)
	ret_code = PVCore::PVLicenseActivator(online_license_path)
	               .online_activation("nobody@testsuite.com", locking_code, "demo_pcap-inspector");
	PV_ASSERT_VALID(ret_code == PVCore::PVLicenseActivator::EError::UNKOWN_USER);

	// Registered but not yet validated mail (USER_NOT_VALIDATED)
	ret_code =
	    PVCore::PVLicenseActivator(online_license_path)
	        .online_activation("unvalidated@testsuite.com", locking_code, "demo_pcap-inspector");
	PV_ASSERT_VALID(ret_code == PVCore::PVLicenseActivator::EError::USER_NOT_VALIDATED);

	// trial license retrieving (TRIAL_ALREADY_ACTIVATED)
	ret_code = PVCore::PVLicenseActivator(online_license_path)
	               .online_activation("validated@testsuite.com", "2222-*222 2222 2222 2222",
	                                  "demo_pcap-inspector");
	PV_ASSERT_VALID(ret_code == PVCore::PVLicenseActivator::EError::TRIAL_ALREADY_ACTIVATED);

	// trial license retrieving (NO_ERROR)
	ret_code = PVCore::PVLicenseActivator(online_license_path)
	               .online_activation("validated@testsuite.com", "1111-*111 1111 1111 1111",
	                                  "demo_pcap-inspector");
	PV_ASSERT_VALID(ret_code == PVCore::PVLicenseActivator::EError::NO_ERROR);

	// trial license generating (NO_ERROR)
	ret_code =
	    PVCore::PVLicenseActivator(online_license_path)
	        .online_activation("validated-generate@testsuite.com", "1401-*1H2 AFZX FK56 VDF8",
	                           "demo_pcap-inspector_dry-run"); // Special locking-code
	PV_ASSERT_VALID(ret_code == PVCore::PVLicenseActivator::EError::NO_ERROR);
}

void test_online_paid_activation()
{
	PVCore::PVLicenseActivator::EError ret_code;

	// Activation service unavailable (ACTIVATION_SERVICE_UNAVAILABLE)
	ret_code = PVCore::PVLicenseActivator(online_license_path)
	               .online_activation("" /*no email*/, "0000-*000 0000 0000 0000",
	                                  "0000000000000000_dry-run");
	PV_ASSERT_VALID(ret_code == PVCore::PVLicenseActivator::EError::ACTIVATION_SERVICE_UNAVAILABLE);

	// Bad activation key (UNKNOWN_ACTIVATION_KEY)
	ret_code = PVCore::PVLicenseActivator(online_license_path)
	               .online_activation("" /*no email*/, locking_code, "InvalidActivaKey");
	PV_ASSERT_VALID(ret_code == PVCore::PVLicenseActivator::EError::UNKNOWN_ACTIVATION_KEY);

	// Already activated license (ACTIVATION_KEY_ALREADY_ACTIVATED)
	ret_code =
	    PVCore::PVLicenseActivator(online_license_path)
	        .online_activation("" /*no email*/, "2222-*222 2222 2222 2222", "1111111111111111");
	PV_ASSERT_VALID(ret_code ==
	                PVCore::PVLicenseActivator::EError::ACTIVATION_KEY_ALREADY_ACTIVATED);

	// Valid activation key retrieving (NO_ERROR)
	ret_code =
	    PVCore::PVLicenseActivator(online_license_path)
	        .online_activation("" /*no email*/, "1111-*111 1111 1111 1111", "1111111111111111");
	PV_ASSERT_VALID(ret_code == PVCore::PVLicenseActivator::EError::NO_ERROR);

	// Valid activation key generation (NO_ERROR)
	ret_code = PVCore::PVLicenseActivator(online_license_path)
	               .online_activation("" /*no email*/, "1401-*1H2 AFZX FK56 VDF8",
	                                  "0000000000000000_dry-run");
	PV_ASSERT_VALID(ret_code == PVCore::PVLicenseActivator::EError::NO_ERROR);
}

void test_offline_activation()
{
	PVCore::PVLicenseActivator::EError ret_code;

	// NO_ERROR
	QFile(offline_license_path).remove();
	ret_code =
	    PVCore::PVLicenseActivator(offline_license_path).offline_activation(online_license_path);
	PV_ASSERT_VALID(ret_code == PVCore::PVLicenseActivator::EError::NO_ERROR);

	// UNABLE_TO_READ_LICENSE_FILE
	ret_code = PVCore::PVLicenseActivator(offline_license_path)
	               .offline_activation("/tmp/unexisting_license.lic");
	PV_ASSERT_VALID(ret_code == PVCore::PVLicenseActivator::EError::UNABLE_TO_READ_LICENSE_FILE);

	// UNABLE_TO_INSTALL_LICENSE_FILE
	chmod(offline_license_folder.toStdString().c_str(), 0); // disable write permissions
	ret_code =
	    PVCore::PVLicenseActivator(offline_license_path).offline_activation(online_license_path);
	chmod(offline_license_folder.toStdString().c_str(), 0775); // reset permissions
	PV_ASSERT_VALID(ret_code == PVCore::PVLicenseActivator::EError::UNABLE_TO_INSTALL_LICENSE_FILE);
}

void cleanup()
{
	PVCore::PVDirectory::remove_rec(online_license_folder);
	PVCore::PVDirectory::remove_rec(offline_license_folder);
}

int main()
{
	atexit(cleanup);

	test_online_trial_activation();
	test_online_paid_activation();
	test_offline_activation();

	return 0;
}

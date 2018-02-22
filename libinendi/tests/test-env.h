/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <cstdlib>
#include <inendi/common.h>
#include <pvbase/general.h>
#include <pvkernel/core/inendi_intrin.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>

#include <License.h>
#include <QDir>

void init_env()
{
	PVCore::PVIntrinsics::init_cpuid();
	setenv("PVKERNEL_PLUGIN_PATH", INENDI_BUILD_DIRECTORY "/libpvkernel/plugins", 0);
	setenv("INENDI_PLUGIN_PATH", INENDI_BUILD_DIRECTORY "/libinendi/plugins", 0);

	static const QString inendi_license_path =
	    QString(INENDI_LICENSE_PATH).replace(0, 1, QDir::homePath());
	Inendi::Utils::License::Init(inendi_license_path.toStdString().c_str());

	PVFilter::PVPluginsLoad::load_all_plugins(); // Splitters
	PVRush::PVPluginsLoad::load_all_plugins();   // Sources

	Inendi::common::load_filters();

	Inendi::Utils::License::Deinit();
}

/* Test case for ticket 1.
 * Here, we try to reproduce this bug by reading a lot of format consecutively, and checking if,
 * for the regexp, the correct one is choosen. This is due to th fact that thread-local objects are used
 * the regexp object, so that it cnap ersist between different formats.
 */

#include <pvcore/PVMeanValue.h>

#include <pvrush/PVFormat.h>
#include <pvrush/PVUnicodeSource.h>
#include <pvrush/PVInputFile.h>
#include <pvrush/PVSourceCreatorFactory.h>
#include <pvrush/PVPluginsLoad.h>

#include <pvfilter/PVPluginsLoad.h>
#include <pvcore/PVArgument.h>

#include <QCoreApplication>
#include <QDir>
#include <QStringList>

#include <iostream>

#include "helpers.h"
#include "test-env.h"

int main(int argc, char** argv)
{
	// Initialisation
	init_env();
	PVFilter::PVPluginsLoad::load_all_plugins();
	PVRush::PVPluginsLoad::load_all_plugins();
	QCoreApplication app(argc, argv);

	// Format reading
	QDir dir_files("test-files/tickets/1/");
	QStringList files = dir_files.entryList(QStringList() << QString("*.format"), QDir::Files | QDir::Readable, QDir::Name);

	// Text file source creator
	PVRush::PVSourceCreator_p text_file_lib = LIB_CLASS(PVRush::PVSourceCreator)::get().get_class_by_name("text_file");
	if (!text_file_lib) {
		std::cerr << "Unable to load the text_file pvrush plugin." << std::endl;
		return 1;
	}

	if (files.size() == 0) {
		std::cerr << "No files in test-files/tickets/1/. Test failed." << std::endl;
		return 1;
	}

	for (int i = 0; i < files.size(); i++) {
		QFileInfo fiformat(files[i]);
		QString file_load = dir_files.absoluteFilePath(fiformat.completeBaseName());

		PVLOG_INFO("Test file %s...\n", qPrintable(file_load));

		bool ok = false;
		// Check the result
		for (int j = 0; j < files.size(); j++) {
			QString const& formatpath = files[j];
			QString fpath = dir_files.absoluteFilePath(formatpath);
			PVRush::PVFormat format("format", fpath);

			float sr = PVRush::PVSourceCreatorFactory::discover_input(PVRush::pair_format_creator(format, text_file_lib), PVCore::PVArgument(file_load));
			if (sr > 0.8 && files[j] == files[i]) {
				ok = true;
			}
		}
		if (!ok) {
			std::cerr << "Format discovery failed !\n" << std::endl;
			return 1;
		}
	}

	return 0;
}

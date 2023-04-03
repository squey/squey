//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVConfig.h> // for PVConfig
#include <pvkernel/core/PVLogger.h> // for PVLOG_INFO, PVLOG_ERROR

#include <pvbase/general.h> // for PVCORE_DIRECTORY_SEP

#include <QDir>
#include <QLibrary>
#include <QSettings>
#include <QString>
#include <QStringList>

// Helper class to load external plugins

// Register function type
using register_class_func = void (*)();
#define register_class_func_string "register_class"

bool PVCore::PVClassLibraryLibLoader::load_class(QString const& path)
{
	QLibrary lib(path);
	register_class_func rf = lib.resolve(register_class_func_string);
	if (rf == nullptr) {
		PVLOG_ERROR("Error while loading plugin %s: %s\n", qPrintable(path),
		            qPrintable(lib.errorString()));
		return false;
	}

	// Call the registry function
	rf();

	// And that's it !
	return true;
}

int PVCore::PVClassLibraryLibLoader::load_class_from_dir(QString const& pluginsdir,
                                                         QString const& prefix)
{
	QDir dir(pluginsdir);
	PVLOG_INFO("Reading plugins directory: %s\n", qPrintable(pluginsdir));

	// Set directory listing filters
	QStringList filters;
	filters << QString("*") + prefix + QString("_*.so");
	dir.setNameFilters(filters);

	QStringList files = dir.entryList();
	QStringListIterator filesIterator(files);
	int count = 0;

	QSettings& pvconfig = PVCore::PVConfig::get().config();
	while (filesIterator.hasNext()) {
		QString curfile = filesIterator.next();
		QString activated_token = curfile + QString("/activated");
		int activated = pvconfig.value(activated_token, 1).toInt();
		if (activated != 0) {
			if (load_class(dir.absoluteFilePath(curfile))) {
				PVLOG_INFO("Successfully loaded plugin '%s'\n", qPrintable(curfile));
				count++;
			}
		}
	}
	return count;
}

int PVCore::PVClassLibraryLibLoader::load_class_from_dirs(QString const& pluginsdirs,
                                                          QString const& prefix)
{
	QStringList pluginsdirs_list = split_plugin_dirs(pluginsdirs);

	int count = 0;
	for (auto & i : pluginsdirs_list) {
		count += load_class_from_dir(i, prefix);
	}
	return count;
}

QStringList PVCore::PVClassLibraryLibLoader::split_plugin_dirs(QString const& dirs)
{
	return dirs.split(PVCORE_DIRECTORY_SEP);
}

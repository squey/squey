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

#include <pvkernel/rush/PVPluginsLoad.h>

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVLogger.h>

#include <QString>
#include <QDir>

#include <cstdlib>

int PVRush::PVPluginsLoad::load_all_plugins()
{
	int ret = 0;
	ret += load_input_type_plugins();
	ret += load_source_plugins();

	return ret;
}

int PVRush::PVPluginsLoad::load_input_type_plugins()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(get_input_type_dir(),
	                                                                INPUT_TYPE_PREFIX);
	if (ret == 0) {
		PVLOG_WARN("No input plugin have been loaded !\n");
	} else {
		PVLOG_INFO("%d input plugins have been loaded.\n", ret);
	}
	return ret;
}

int PVRush::PVPluginsLoad::load_source_plugins()
{
	int ret =
	    PVCore::PVClassLibraryLibLoader::load_class_from_dirs(get_source_dir(), SOURCE_PREFIX);
	if (ret == 0) {
		PVLOG_WARN("No source plugin have been loaded !\n");
	} else {
		PVLOG_INFO("%d source plugins have been loaded.\n", ret);
	}
	return ret;
}

QString PVRush::PVPluginsLoad::get_input_type_dir()
{
	QString pluginsdirs = QString(getenv("PVKERNEL_PLUGIN_PATH"));
	if (pluginsdirs.isEmpty()) {
		pluginsdirs = QString(PVKERNEL_PLUGIN_PATH);
	}

	return pluginsdirs + QDir::separator() + "input-types";
}

QString PVRush::PVPluginsLoad::get_source_dir()
{
	QString pluginsdirs = QString(getenv("PVKERNEL_PLUGIN_PATH"));
	if (pluginsdirs.isEmpty()) {
		pluginsdirs = QString(PVKERNEL_PLUGIN_PATH);
	}

	return pluginsdirs + QDir::separator() + "sources";
}

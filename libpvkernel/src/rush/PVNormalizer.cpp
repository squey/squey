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

#include <pvkernel/rush/PVNormalizer.h>

#include <pvbase/general.h> // for PVCORE_DIRECTORY_SEP

#include <cstdlib> // for getenv

#include <QDir>
#include <QString>
#include <QStringList>

/******************************************************************************
 *
 * PVRush::normalize_get_helpers_plugins_dirs
 *
 *****************************************************************************/
QStringList PVRush::normalize_get_helpers_plugins_dirs(QString helper)
{
	QString pluginsdirs(std::getenv("PVFORMAT_HELPER"));
	if (pluginsdirs.isEmpty()) {
		pluginsdirs = QString(INENDI_CONFIG);
	}
	pluginsdirs += "/normalize-helpers";

	QStringList pluginsdirs_list = pluginsdirs.split(PVCORE_DIRECTORY_SEP);
	for (int counter = 0; counter < pluginsdirs_list.count(); counter++) {
		if (pluginsdirs_list[counter].startsWith("~/")) {
			pluginsdirs_list[counter].replace(0, 1, QDir::homePath());
		}
		pluginsdirs_list[counter] =
		    pluginsdirs_list[counter] + QDir::separator() + helper + QDir::separator();
	}

	return pluginsdirs_list;
}

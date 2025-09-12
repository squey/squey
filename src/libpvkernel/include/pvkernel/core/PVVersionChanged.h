/* * MIT License
 *
 * © Squey, 2025
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __PVKERNEL_PVVERSIONCHANGED_H__
#define __PVKERNEL_PVVERSIONCHANGED_H__

#include <pvkernel/core/PVSingleton.h>
#include <pvkernel/core/PVConfig.h>

#include <QDir>
#include <QFile>
#include <QTextStream>

#include SQUEY_VERSION_FILE_PATH

namespace PVCore
{

class PVVersionChanged
{
public:
  PVVersionChanged()
  {
    QString current_version = QString(SQUEY_CURRENT_VERSION_STR);
    QString previous_version;

    QFile version_file(PVCore::PVConfig::user_dir() + QDir::separator() + "version.txt");
    if (version_file.open(QFile::ReadOnly | QFile::Text)) {
    	QTextStream in(&version_file);
    	previous_version = in.readLine();
    	version_file.close();
    }

    if (current_version != previous_version) {
       	_version_changed = true;
    }

    if (version_file.open(QIODevice::WriteOnly)) {
    	QTextStream out(&version_file);
    	out << current_version;
    }
  }

public:
    static bool version_changed()
    {
        return PVSingleton<PVVersionChanged>::get()._version_changed;
    }

  private:
    bool _version_changed = false;
};

} // namespace PVCore

#endif // __PVKERNEL_PVVERSIONCHANGED_H__

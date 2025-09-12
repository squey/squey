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
#pragma once

#include <QDir>
#include <QFileInfoList>
#include <QStandardPaths>

#include <boost/dll/runtime_symbol_info.hpp>

#include <pvkernel/core/PVArchive.h>
#include <pvkernel/core/PVVersionChanged.h>

namespace PVCore
{

class PVUtilitiesDecompressor
{
public:
    PVUtilitiesDecompressor()
    {
        if (not PVCore::PVVersionChanged::version_changed()) {
            return;
        }

        const QString& app_cache = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
        const std::string& appdir = boost::dll::program_location().parent_path().string();

        const QFileInfoList& files = QDir(QString::fromStdString(appdir)).entryInfoList({ "*.zip" }, QDir::Files);
        for (const QFileInfo& file_info : files) {
            const QString& output_dir = QDir(app_cache).filePath(file_info.completeBaseName());
            QDir(output_dir).removeRecursively();
            QStringList extracted_files;
            PVCore::PVArchive::extract(file_info.absoluteFilePath(), output_dir, extracted_files);
        }
    }
};

} // namespace PVCore

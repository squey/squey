/* * MIT License
 *
 * © Squey, 2026
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

// That a directory counts as writable when a directory can be created in it, as an
// import creates its collection there: QFileInfo::isWritable() only reads the
// read-only attribute on Windows, and calls a writable file writable.

#include <pvkernel/core/PVUtils.h>
#include <pvkernel/core/squey_assert.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>

int main()
{
	QTemporaryDir scratch;
	PV_ASSERT_VALID(scratch.isValid(), "no scratch directory", 0);

	PV_VALID(PVCore::can_create_directory_in(scratch.path()), true);
	// The directory it creates to know is gone.
	PV_VALID(QDir(scratch.path()).isEmpty(), true);

	const QString file = scratch.filePath("file");
	{
		QFile created(file);
		PV_ASSERT_VALID(created.open(QIODevice::WriteOnly), "could not write", file.toStdString());
	}
	PV_VALID(QFileInfo(file).isWritable(), true);
	PV_VALID(PVCore::can_create_directory_in(file), false);

	PV_VALID(PVCore::can_create_directory_in(scratch.filePath("missing")), false);
	// Not the current directory.
	PV_VALID(PVCore::can_create_directory_in(QString()), false);

	return 0;
}

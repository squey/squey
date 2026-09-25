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

// That removing a directory removes what is in it, and nothing it points to.
//
// remove_rec() asked every entry whether it was a directory, which for a link
// answers about the target, and walked into it: a link inside a directory being
// cleared -- the collections directory a run clears on its way in and out --
// emptied whatever it pointed at, wherever that was. A link goes as a link.

#include <pvkernel/core/PVDirectory.h>
#include <pvkernel/core/squey_assert.h>

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QTemporaryDir>

namespace
{

// On Windows a link is a shortcut, whose name has to carry the extension.
#ifdef _WIN32
constexpr const char* link_suffix = ".lnk";
#else
constexpr const char* link_suffix = "";
#endif

void touch(const QString& path)
{
	QFile file(path);
	PV_ASSERT_VALID(file.open(QIODevice::WriteOnly), "could not write", path.toStdString());
	file.write("kept");
}

} // namespace

int main()
{
	QTemporaryDir scratch;
	PV_ASSERT_VALID(scratch.isValid(), "no scratch directory", 0);

	// What a link in the tree points at, and must survive its removal.
	const QString outside = scratch.path() + "/outside";
	QDir().mkpath(outside + "/deeper");
	touch(outside + "/kept.txt");
	touch(outside + "/deeper/kept.txt");

	const QString tree = scratch.path() + "/tree";
	QDir().mkpath(tree + "/sub");
	touch(tree + "/file.txt");
	touch(tree + "/sub/file.txt");
	PV_ASSERT_VALID(QFile::link(outside, tree + "/to_directory" + link_suffix),
	                "could not link to a directory", 0);
	PV_ASSERT_VALID(QFile::link(outside + "/kept.txt", tree + "/sub/to_file" + link_suffix),
	                "could not link to a file", 0);

	PV_VALID(PVCore::PVDirectory::remove_rec(tree), true);

	PV_VALID(QFileInfo::exists(tree), false);
	PV_VALID(QFileInfo::exists(outside + "/kept.txt"), true);
	PV_VALID(QFileInfo::exists(outside + "/deeper/kept.txt"), true);

	return 0;
}

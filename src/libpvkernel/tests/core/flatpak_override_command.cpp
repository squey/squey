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

// That the command the Flatpak gives to let it reach a directory of the host names that
// directory to flatpak once a shell has read it, whatever the characters of its path.

#include <pvkernel/core/PVFlatpakHelper.h>
#include <pvkernel/core/squey_assert.h>

#include <QtGlobal>

#include <string>

static std::string command(QString const& directory)
{
	return PVCore::PVFlatpakHelper::filesystem_override_command(directory).toStdString();
}

int main()
{
	qputenv("FLATPAK_ID", "org.squey.Squey");

	PV_VALID(command("/srv/tmp-squey"),
	         std::string("flatpak override --user --filesystem=/srv/tmp-squey org.squey.Squey"));
	PV_VALID(command("/srv/my data"),
	         std::string("flatpak override --user '--filesystem=/srv/my data' org.squey.Squey"));
	PV_VALID(command("/srv/it's"),
	         std::string(R"(flatpak override --user '--filesystem=/srv/it'\''s' org.squey.Squey)"));
	// Flatpak would take what follows a bare colon for the access mode.
	PV_VALID(command("/srv/a:b"),
	         std::string(R"(flatpak override --user '--filesystem=/srv/a\:b' org.squey.Squey)"));
	PV_VALID(command(R"(/srv/a\b)"),
	         std::string(R"(flatpak override --user '--filesystem=/srv/a\\b' org.squey.Squey)"));

	return 0;
}

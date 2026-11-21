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

#ifndef __PVKERNEL_PVFLATPAKHELPER_H__
#define __PVKERNEL_PVFLATPAKHELPER_H__

#include <QFileInfo>
#include <QString>

namespace PVCore
{

/** The Flatpak sandbox, where Squey reaches the files of the user through the portals alone. */
class PVFlatpakHelper
{
  public:
	static bool is_flatpak()
	{
		// Bind-mounted into every Flatpak sandbox, and what Qt looks for to use the portals.
		static const bool flatpak = QFileInfo::exists("/.flatpak-info");
		return flatpak;
	}

	/**
	 * Whether @p path is one of the files and directories the portals hand the sandbox over.
	 *
	 * The document portal serves them from a file system of its own, which only shows the
	 * documents themselves: a file written beside one of them stays out of sight of the host.
	 */
	static bool is_document_portal_path(QString const& path);

	/**
	 * Where @p path is on the host, for a document of the portal, which the sandbox can only
	 * show; @p path itself otherwise, or when the portal does not tell. Asks the portal once per
	 * document, blocking for a second at most.
	 */
	static QString host_path(QString const& path);

	/**
	 * Where the sandbox reaches @p path, a directory of the portal, without the portal: the
	 * directory of the host itself, when a permission of the sandbox, such as the one
	 * filesystem_override_command() grants, puts it in sight; @p path otherwise, and for any
	 * other path.
	 *
	 * The portal hands such a directory over through its file system all the same when several
	 * branches of the application are installed:
	 * https://github.com/flatpak/xdg-desktop-portal/issues/879. Creates a directory in it, then
	 * removes it, to tell it from a directory the sandbox has of its own at the same path, as
	 * under /tmp. Blocks as host_path() does.
	 */
	static QString direct_directory(QString const& path);

	/**
	 * The command that lets the sandbox reach @p host_directory, a directory of the host, from
	 * its next start on, quoted for a shell.
	 */
	static QString filesystem_override_command(QString const& host_directory);
};

} // namespace PVCore

#endif // __PVKERNEL_PVFLATPAKHELPER_H__

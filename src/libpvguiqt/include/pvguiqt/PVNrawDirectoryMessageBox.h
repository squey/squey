/* * MIT License
 *
 * © ESI Group, 2015
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

#ifndef __PVGUIQT_PVNRAWDIRECTORYMESSAGEBOX_H__
#define __PVGUIQT_PVNRAWDIRECTORYMESSAGEBOX_H__

#include <QClipboard>
#include <QDir>
#include <QFileInfo>
#include <QGuiApplication>
#include <QMessageBox>
#include <QPushButton>

#include <pvkernel/core/PVFlatpakHelper.h>
#include <pvkernel/core/PVKernelSettings.h>
#include <pvkernel/core/PVSettings.h>
#include <pvkernel/core/PVUtils.h>
#include <pvkernel/widgets/PVFileDialog.h>

class PVNrawDirectoryMessageBox : public QMessageBox
{
  public:
	PVNrawDirectoryMessageBox(QWidget* parent = nullptr) : QMessageBox(parent)
	{
		QString nraw_tmp = PVCore::settings::working_directory.value();
		const QString default_directory =
		    PVCore::PVSettings::get()
		        .default_value(PVCore::settings::working_directory.key())
		        .toString();

		// A directory chosen through the portal that the sandbox has since been let reach
		// directly, by the command explain() gives.
		if (const QString direct = PVCore::PVFlatpakHelper::direct_directory(nraw_tmp);
		    direct != nraw_tmp) {
			PVCore::settings::working_directory.set_value(direct);
			nraw_tmp = PVCore::settings::working_directory.value();
		}

		// The directory of the sandbox is there for Squey alone, and the other ones it can only
		// reach through the portal, slowly.
		if (PVCore::PVFlatpakHelper::is_flatpak() and nraw_tmp == default_directory) {
			QDir().mkpath(nraw_tmp);
		}

		if (PVCore::can_create_directory_in(nraw_tmp)) {
			return;
		}

		setStandardButtons(QMessageBox::Cancel);
		setIcon(QMessageBox::Information);
		QAbstractButton* default_button = nullptr;
		if (nraw_tmp == default_directory) {
			setWindowTitle("Welcome to Squey!");
			setText(QString("It seems that you are using Squey for the first time.<br><br>"
			                "Please, chose a temporary files directory for Squey.<br>"
			                "A fast underlying storage for this directory will provide better "
			                "performances."));
		} else {
			// A directory set before, out of reach, such as one of the host for a Flatpak that
			// no longer sees it.
			setWindowTitle("Working directory");
			setText(QString("Squey cannot write in its working directory, <i>%1</i>.<br><br>"
			                "Please, choose another one.<br>"
			                "A fast underlying storage for this directory will provide better "
			                "performances.")
			            .arg(nraw_tmp.toHtmlEscaped()));
			QDir().mkpath(default_directory);
			if (PVCore::can_create_directory_in(default_directory)) {
				setInformativeText(QString("The default one is <i>%1</i>.")
				                       .arg(default_directory.toHtmlEscaped()));
				default_button = addButton(tr("Use the default"), QMessageBox::ActionRole);
			}
		}

		QAbstractButton* browser_button = addButton(tr("Browse..."), QMessageBox::ActionRole);

		do {
			exec();

			if (clickedButton() == browser_button) {
				const QString chosen = PVWidgets::PVFileDialog::getExistingDirectory();
				if (not chosen.isEmpty()) {
					PVCore::settings::working_directory.set_value(
					    working_directory_for(this, chosen));
					// The administrator may have set it.
					nraw_tmp = PVCore::settings::working_directory.value();
				}
			} else if (default_button != nullptr and clickedButton() == default_button) {
				PVCore::PVSettings::get().reset(PVCore::settings::working_directory.key());
				nraw_tmp = PVCore::settings::working_directory.value();
			} else if (clickedButton() == button(QMessageBox::Cancel)) {
				::exit(0);
			}

		} while (not PVCore::can_create_directory_in(nraw_tmp));
	}

	/** How Squey reaches a working directory. */
	enum class reach {
		/** It writes there directly. */
		direct,
		/** Through the portal, slowly, the Flatpak having no permission of its own for it. */
		through_portal,
		/** Not at all: the Flatpak does not see this directory of the host. */
		out_of_sight,
		/** Not at all, for another reason. */
		out_of_reach
	};

	/**
	 * How Squey reaches @p directory. Creates a directory in it, then removes it, to know, and
	 * blocks as PVFlatpakHelper::host_path() does for a directory of the portal.
	 */
	static reach reach_of(QString const& directory)
	{
		const bool portal = PVCore::PVFlatpakHelper::is_document_portal_path(directory);
		if (not PVCore::can_create_directory_in(directory)) {
			return PVCore::PVFlatpakHelper::is_flatpak() and not portal and
			               QDir::isAbsolutePath(directory) and not QFileInfo::exists(directory)
			           ? reach::out_of_sight
			           : reach::out_of_reach;
		}
		return portal and PVCore::PVFlatpakHelper::direct_directory(directory) == directory
		           ? reach::through_portal
		           : reach::direct;
	}

	/**
	 * Says why Squey reaches @p directory through the portal, or does not see it, as @p how
	 * tells, and gives the command that lets it reach the directory directly. Does nothing for
	 * another reach.
	 */
	static void explain(QWidget* parent, QString const& directory, reach how)
	{
		if (how != reach::through_portal and how != reach::out_of_sight) {
			return;
		}
		const QString host = how == reach::through_portal
		                         ? PVCore::PVFlatpakHelper::host_path(directory)
		                         : directory;
		// A portal older than GetHostPaths does not say where the directory is.
		const bool host_known = how == reach::out_of_sight or host != directory;
		const QString command = PVCore::PVFlatpakHelper::filesystem_override_command(
		    host_known ? host : QString("DIRECTORY"));
		QString title;
		QString text;
		if (how == reach::through_portal) {
			title = tr("Slow working directory");
			text =
			    tr("Squey reaches %1 through the portal of the desktop, which serves each first "
			       "read of its data, several times slower than the disk does.<br><br>"
			       "To let Squey reach it directly, run"
			       "<pre>%2</pre>"
			       "%3")
			        .arg(host_known ? "<i>" + host.toHtmlEscaped() + "</i>" : tr("this directory"),
			             command.toHtmlEscaped(),
			             host_known ? tr("then restart Squey: there is no need to choose the "
			                             "directory again.")
			                        : tr("with the path of the directory on the host for "
			                             "DIRECTORY, then restart Squey and choose the "
			                             "directory again."));
		} else {
			title = tr("Working directory out of sight");
			text = tr("The Flatpak of Squey does not see <i>%1</i>: it only sees the directories "
			          "it is given.<br><br>"
			          "If the directory exists, give it to Squey with"
			          "<pre>%2</pre>"
			          "then restart Squey.")
			           .arg(host.toHtmlEscaped(), command.toHtmlEscaped());
		}
		QMessageBox box(QMessageBox::Warning, title, text, QMessageBox::Close, parent);
		box.setTextInteractionFlags(Qt::TextSelectableByMouse);
		QPushButton* copy = box.addButton(tr("Copy the command"), QMessageBox::AcceptRole);
		box.setDefaultButton(copy);
		box.exec();
		if (box.clickedButton() == copy) {
			QGuiApplication::clipboard()->setText(command);
		}
	}

	/**
	 * The working directory to set for @p chosen, a directory the user chose: the one
	 * PVFlatpakHelper::direct_directory() gives. Warns of a directory the sandbox only reaches
	 * through the portal, and says how to let it reach the directory directly.
	 */
	static QString working_directory_for(QWidget* parent, QString const& chosen)
	{
		const QString directory = PVCore::PVFlatpakHelper::direct_directory(chosen);
		if (PVCore::PVFlatpakHelper::is_document_portal_path(directory)) {
			explain(parent, directory, reach::through_portal);
		}
		return directory;
	}
};

#endif // __PVGUIQT_PVNRAWDIRECTORYMESSAGEBOX_H__

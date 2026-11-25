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

#ifndef PCAP_GUI_PROFILESPANEL_H
#define PCAP_GUI_PROFILESPANEL_H

#include <QWidget>

#include <functional>

class QAction;
class QListWidget;
class QListWidgetItem;
class QMenu;
class QToolButton;

namespace PVPcapsicum
{

class ProfileEditor;

/**
 * The profiles to choose from, those Squey ships then those of the user, with what they
 * import: created, duplicated, renamed in place with F2, deleted, imported from files and
 * exported to files.
 */
class ProfilesPanel : public QWidget
{
	Q_OBJECT

  public:
	explicit ProfilesPanel(ProfileEditor* editor, QWidget* parent = nullptr);

	//! New profile, Duplicate, Rename, Delete, Import, Export.
	QMenu* actions_menu() const { return _menu; }

	void create();
	void duplicate();
	void rename();
	void remove();
	void import();
	void export_profile();

  private:
	void refresh();
	void select_current();
	void renamed(QListWidgetItem* item);
	//! Runs what changes profiles, telling the user why it could not.
	bool attempt(const QString& what, const std::function<void()>& change);

	ProfileEditor* _editor;
	QListWidget* _list;
	QMenu* _menu;
	QAction* _rename;
	QAction* _remove;
	bool _refreshing = false;
};

} // namespace PVPcapsicum

#endif // PCAP_GUI_PROFILESPANEL_H

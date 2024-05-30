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

#ifndef PVROOTTREEVIEW_H
#define PVROOTTREEVIEW_H

#include <QTreeView>

namespace PVCore
{
class PVDataTreeObject;
} // namespace PVCore

namespace Squey
{
class PVScaled;
class PVMapped;
class PVView;
} // namespace Squey

namespace PVGuiQt
{

class PVRootTreeModel;

class PVRootTreeView : public QTreeView
{
	Q_OBJECT

  public:
	explicit PVRootTreeView(QAbstractItemModel* model, QWidget* parent = 0);

  protected:
	void mouseDoubleClickEvent(QMouseEvent* event) override;
	void contextMenuEvent(QContextMenuEvent* event) override;
	void enterEvent(QEnterEvent* event) override;
	void leaveEvent(QEvent* event) override;

  protected Q_SLOTS:
	// Actions slots
	void create_new_view();
	void edit_mapping();
	void edit_scaling();

  protected:
	PVCore::PVDataTreeObject* get_selected_obj();

	template <typename T>
	T* get_selected_obj_as()
	{
		return dynamic_cast<T*>(this->get_selected_obj());
	}

  protected:
	PVRootTreeModel* tree_model();
	PVRootTreeModel const* tree_model() const;

  private:
	QAction* _act_new_view;
	QAction* _act_new_scaled;
	QAction* _act_new_mapped;
	QAction* _act_edit_mapping;
	QAction* _act_edit_scaling;
};
} // namespace PVGuiQt

#endif // PVROOTTREEVIEW_H

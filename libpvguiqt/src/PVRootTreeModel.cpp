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

#include <squey/PVRoot.h>
#include <squey/PVView.h>

#include <pvguiqt/PVRootTreeModel.h>

#include <QBrush>
#include <QFont>

PVGuiQt::PVRootTreeModel::PVRootTreeModel(Squey::PVSource& root, QObject* parent)
    : PVHiveDataTreeModel(root, parent)
{
}

QVariant PVGuiQt::PVRootTreeModel::data(const QModelIndex& index, int role) const
{
	if (auto* v =
	        dynamic_cast<Squey::PVView*>((PVCore::PVDataTreeObject*)index.internalPointer())) {
		if (role == Qt::FontRole) {
			if (v->get_parent<Squey::PVRoot>().current_view() == v) {
				QFont font;
				font.setBold(true);
				return font;
			}
		} else if (role == Qt::ForegroundRole) {
			return QBrush(v->get_color());
		}
	}

	return PVHiveDataTreeModel::data(index, role);
}

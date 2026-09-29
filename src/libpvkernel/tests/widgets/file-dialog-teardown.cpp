//
// MIT License
//
// © Squey, 2026
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

// A file dialog closed while its views hold an index its model lost track of.
//
// On Windows, QFileSystemModel can move the persistent indexes of a directory it renamed to its
// case on disk to row -1, out of its table but still pointing at it: a model destroyed before the
// views holding such an index is read again, freed, as they release it. Squey crashed that way,
// closing the dialog in which a first launch chooses its temporary directory.

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/widgets/PVFileDialog.h>

#include <QAbstractItemModel>
#include <QApplication>
#include <QFileDialog>
#include <QPointer>
#include <QTemporaryDir>
#include <QTreeView>

namespace
{

/**
 * Does to the persistent indexes of @p index what QFileSystemModel::sort() does to those of a
 * directory it no longer finds among the visible children of its parent.
 */
void lose_track_of(const QModelIndex& index)
{
	// Only a model may change its persistent indexes.
	struct Model : QAbstractItemModel {
		using QAbstractItemModel::changePersistentIndex;
		using QAbstractItemModel::createIndex;
	};
	using CreateIndex = QModelIndex (QAbstractItemModel::*)(int, int, const void*) const;
	constexpr auto create_index = static_cast<CreateIndex>(&Model::createIndex);

	auto* model = const_cast<QAbstractItemModel*>(index.model());
	(model->*&Model::changePersistentIndex)(
	    index, (model->*create_index)(-1, index.column(), index.internalPointer()));
}

} // namespace

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid());

	bool track_lost = false;
	bool model_outlived_views = false;

	// Posted, so that the modal loop of the dialog runs it.
	QMetaObject::invokeMethod(
	    &app,
	    [&] {
		    auto* dialog = qobject_cast<QFileDialog*>(QApplication::activeModalWidget());
		    PV_ASSERT_VALID(dialog != nullptr);
		    QPointer<QTreeView> view = dialog->findChild<QTreeView*>();
		    PV_ASSERT_VALID(not view.isNull());

		    const QModelIndex root = view->rootIndex();
		    PV_ASSERT_VALID(root.isValid());
		    QObject::connect(root.model(), &QObject::destroyed, [&model_outlived_views, view] {
			    model_outlived_views = view.isNull();
		    });

		    lose_track_of(root);
		    track_lost = not view->rootIndex().isValid() and view->rootIndex().model() != nullptr;

		    dialog->reject();
	    },
	    Qt::QueuedConnection);

	PVWidgets::PVFileDialog::getExistingDirectory(nullptr, {}, dir.path());

	PV_ASSERT_VALID(track_lost, "the root index of the views", "still tracked by the model");
	PV_ASSERT_VALID(model_outlived_views, "the file system model", "destroyed before the views");

	return 0;
}

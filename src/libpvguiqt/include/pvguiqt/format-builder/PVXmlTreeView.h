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

#ifndef MYTREEVIEW_H
#define MYTREEVIEW_H

#include <QTreeView>
#include <QTreeWidgetItem>

#include <QMouseEvent>

#include <iostream>
#include <PVXmlParamWidget.h>
#include <PVXmlDomModel.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>

// QTreeView

const int ExtraHeight = 3;

namespace PVRush
{
class PVXmlTreeNodeDom;
} // namespace PVRush

namespace App
{

class PVXmlTreeView : public QTreeView /* public QAbstractItemView*/
{
	Q_OBJECT

  public:
	enum AddType { addRegEx, addFilter, addAxis, addUrl };
	PVXmlTreeView(QWidget* parent = nullptr);
	~PVXmlTreeView() override;

	void addAxisIn();

	/**
	 * Add a new Filter after the selected element.
	 */
	void addFilterAfter();

	/**
	 * add a new splitter in DOM refering to the splitter plugin
	 * @param splitterPlugin : new instance of the plugin requesting the new splitter
	 */
	PVRush::PVXmlTreeNodeDom* addSplitter(PVFilter::PVFieldsSplitterParamWidget_p splitterPlugin);
	PVRush::PVXmlTreeNodeDom*
	addConverter(PVFilter::PVFieldsConverterParamWidget_p converterPlugin);

	PVRush::PVXmlTreeNodeDom* processChildrenWithField();

	/**
	 * Add a new RegEx after the selected element.
	 */
	void addRegExIn();
	void addUrlIn();

	void addNode(AddType type);

	void applyModification(PVXmlParamWidget* paramBord, QModelIndex& index);

	/**
	 * Delete the selected Item
	 */
	void deleteSelection();

	/**
	 * expand recursively fields.
	 * @param index
	 */
	void expandRecursive(const QModelIndex& index);

	void mouseDoubleClickEvent(QMouseEvent* event) override;

	PVXmlDomModel* getModel();

	QModelIndex getInsertionIndex()
	{
		QModelIndexList lsel = selectedIndexes();

		if (lsel.count() == 0) {
			return QModelIndex();
		} else {
			return lsel[0];
		}
	}

	void postInsertion(const QModelIndex& index, PVRush::PVXmlTreeNodeDom* node)
	{
		if (node) {
			QModelIndex child = getModel()->indexOfChild(index, node);
			setCurrentIndex(child);
			Q_EMIT clicked(child);
		}
	}

	void refresh() { Q_EMIT clicked(currentIndex()); }

	/**
	 * Move down the selected element.
	 */
	void moveDown();

	/**
	 * Move up the selected element.
	 */
	void moveUp();

  protected:
  private:
	bool isDraging;
	bool isEditing;
  public Q_SLOTS:
	void slotDataHasChanged(const QModelIndex&, const QModelIndex&, const QVector<int>&);
	void slotSelectNext();
};
} // namespace App
#endif /* MYTREEVIEW_H */

/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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

namespace PVInspector
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
		} else if (!lsel[0].parent().isValid()) {
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
	void slotDataHasChanged(const QModelIndex&, const QModelIndex&);
	void slotSelectNext();
  Q_SIGNALS:
	void refresh();
};
} // namespace PVInspector
#endif /* MYTREEVIEW_H */

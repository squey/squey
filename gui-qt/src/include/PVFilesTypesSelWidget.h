/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFILESTYPESELWIDGET_H
#define PVFILESTYPESELWIDGET_H

#include <QtCore>
#include <QDialog>

#include <QTableView>
#include <map>
#include <QAbstractTableModel>
#include <QStyledItemDelegate>
#include <QListWidget>
#include <QCheckBox>
#include <QVariant>

namespace PVInspector
{
class PVMainWindow;

typedef std::map<QString, QStringList> map_files_types;

class PVFilesTypesSelModel : public QAbstractTableModel
{
  public:
	PVFilesTypesSelModel(map_files_types& files_types, QObject* parent = 0);

  public:
	int rowCount(const QModelIndex& parent) const override;
	int columnCount(const QModelIndex& parent) const override;
	QVariant data(const QModelIndex& index, int role) const override;
	bool setData(const QModelIndex& index, const QVariant& value, int role) override;
	Qt::ItemFlags flags(const QModelIndex& index) const override;
	QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
	void emitAllTypesChanged();

  protected:
	map_files_types& _files_types;
	map_files_types _org_files_types;
	QString _header_name[2];
};

class PVFilesTypesSelDelegate : public QStyledItemDelegate
{
  public:
	PVFilesTypesSelDelegate(QObject* parent = 0);

  public:
	QWidget* createEditor(QWidget* parent,
	                      const QStyleOptionViewItem& option,
	                      const QModelIndex& index) const override;
	void setEditorData(QWidget* editor, const QModelIndex& index) const override;
	void setModelData(QWidget* editor,
	                  QAbstractItemModel* model,
	                  const QModelIndex& index) const override;
	void updateEditorGeometry(QWidget* editor,
	                          const QStyleOptionViewItem& option,
	                          const QModelIndex& index) const override;
	QSize sizeHint(const QStyleOptionViewItem& option, const QModelIndex& index) const override;
};

class PVFilesTypesSelWidget : public QDialog
{
	Q_OBJECT

  public:
	PVFilesTypesSelWidget(PVMainWindow* parent, map_files_types& files_types);

  public Q_SLOTS:
	void apply_all();
	void all_types_check_Slot(int state);

  private:
	PVMainWindow* main_window;
	QTableView* _files_types_view;
	QListWidget* _all_types_list;
	PVFilesTypesSelModel* _files_types_model;
	PVFilesTypesSelDelegate* _files_types_del;
	QCheckBox* _all_types_check;
	map_files_types& _files_types;
};
} // namespace PVInspector

#endif // PVFILTERSEARCHWIDGET_H

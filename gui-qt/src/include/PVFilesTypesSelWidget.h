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

#include <picviz/general.h>

namespace PVInspector {
class PVMainWindow;

typedef std::map<QString,QStringList> map_files_types;

class PVFilesTypesSelModel : public QAbstractTableModel {
public:
	PVFilesTypesSelModel(map_files_types& files_types, QObject* parent = 0);
public:
	int rowCount(const QModelIndex &parent) const;
	int columnCount(const QModelIndex &parent) const;
	QVariant data(const QModelIndex& index, int role) const;
	bool setData(const QModelIndex& index, const QVariant &value, int role);
	Qt::ItemFlags flags(const QModelIndex& index) const;
	QVariant headerData(int section, Qt::Orientation orientation, int role) const; 
	void emitAllTypesChanged();
protected:
	map_files_types& _files_types;
	map_files_types  _org_files_types;
	QString          _header_name[2];
};

class PVFilesTypesSelDelegate : public QStyledItemDelegate {
public:
	PVFilesTypesSelDelegate(QObject* parent = 0);
public:
	QWidget* createEditor(QWidget *parent, const QStyleOptionViewItem &option, const QModelIndex &index) const;
	void setEditorData(QWidget* editor, const QModelIndex& index) const;
	void setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const;
	void updateEditorGeometry(QWidget* editor, const QStyleOptionViewItem &option, const QModelIndex &index) const;
	QSize sizeHint(const QStyleOptionViewItem & option, const QModelIndex & index) const;

};

class PVFilesTypesSelWidget : public QDialog
{
	Q_OBJECT

public:
	PVFilesTypesSelWidget(PVMainWindow *parent, map_files_types& files_types);
	
public slots:
	void apply_all();
	void all_types_check_Slot(int state);

private:
	PVMainWindow*            main_window;
	QTableView*              _files_types_view;
	QListWidget*             _all_types_list;
	PVFilesTypesSelModel*    _files_types_model;
	PVFilesTypesSelDelegate* _files_types_del;
	QCheckBox*               _all_types_check;
	map_files_types&         _files_types;
};


}

#endif // PVFILTERSEARCHWIDGET_H




#ifndef PVINSPECTOR_PVLISTCOLNRAWDLG_H
#define PVINSPECTOR_PVLISTCOLNRAWDLG_H

#include <picviz/PVView_types.h>

#include <PVListDisplayDlg.h>

#include <QAbstractListModel>
#include <QVector>
#include <QDialog>

namespace PVRush {
class PVNraw;
}

namespace PVInspector {

namespace __impl {

class PVListColNrawModel: public QAbstractListModel
{
	Q_OBJECT

public:
	PVListColNrawModel(PVRush::PVNraw const* nraw, std::vector<int> const* idxes, size_t nvalues, PVCol nraw_col, QWidget* parent = NULL);

public:
	int rowCount(QModelIndex const& parent = QModelIndex()) const;
	QVariant data(QModelIndex const& index, int role = Qt::DisplayRole) const;
	QVariant headerData(int section, Qt::Orientation orientation, int role) const;

private:
	PVRush::PVNraw const* _nraw;
	PVCol _nraw_col;
	std::vector<int> const* _idxes;
	size_t _nvalues;
};

}

class PVListColNrawDlg: public PVListDisplayDlg
{
	Q_OBJECT

public:
	PVListColNrawDlg(Picviz::PVView const& view, std::vector<int> const& idxes, size_t nvalues, PVCol view_col, QWidget* parent = NULL);
};

}

#endif

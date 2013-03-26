#include <pvkernel/core/general.h>
#include <QTableView>
#include <QAbstractTableModel>
#include <QModelIndex>
#include <QVariant>
#include <QApplication>
#include <QMainWindow>
#include <QHeaderView>

#define NROWS 1000000000 // OK w/ 71580000, not w/ 71590000
#define NCOLS 1

class MyModel: public QAbstractTableModel
{
public:
	int rowCount(QModelIndex const&) const override { return NROWS; }
	int columnCount(QModelIndex const&) const override { return NCOLS; }
	QVariant data(QModelIndex const& idx, int role) const override
	{
		if (role == Qt::DisplayRole) {
			//PVLOG_INFO("data w/ idx %d/%d\n", idx.row(), idx.column());
			return QVariant(QString::number(idx.row()));
		}
		else {
			//PVLOG_INFO("data w/ idx %d/%d and role %d\n", idx.row(), idx.column(), role);
		}
		return QVariant();
	}
};

int main(int argc, char** argv)
{
	QApplication app(argc, argv);
	PVLOG_INFO("Init model + view...\n");
	MyModel* model = new MyModel();
	QTableView* table = new QTableView();
	/*table->horizontalHeader()->setResizeMode(QHeaderView::Fixed);
	table->horizontalHeader()->setStretchLastSection(false);
	table->verticalHeader()->setResizeMode(QHeaderView::Fixed);
	table->verticalHeader()->setStretchLastSection(false);*/
	table->setWordWrap(false);
	table->horizontalHeader()->setStretchLastSection(true);
	table->verticalHeader()->setDefaultSectionSize(table->verticalHeader()->minimumSectionSize());
	table->setModel(model);
	table->setGridStyle(Qt::NoPen);
	table->horizontalHeader()->hide();
	table->setContextMenuPolicy(Qt::ActionsContextMenu);
	table->verticalHeader()->hide();

	PVLOG_INFO("Done. Creating main window and showing...\n");
	QMainWindow* mw = new QMainWindow();
	mw->setCentralWidget(table);
	mw->show();

	PVLOG_INFO("Run Qt app..\n");

	return app.exec();
}

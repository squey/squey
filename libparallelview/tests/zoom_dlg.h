#ifndef ZOOMDLG_H
#define ZOOMDLG_H

#include <pvparallelview/PVZonesDrawing.h>
#include <pvparallelview/PVZoomedParallelScene.h>
#include <pvparallelview/PVZoomedParallelView.h>

#include <QDialog>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>

class ZoomDlg: public QDialog
{
public:
	ZoomDlg(PVParallelView::PVZonesDrawing<PARALLELVIEW_ZZT_BBITS>& zd, QWidget* parent = NULL):
		_zd(zd)
	{
		_zedit = new QLineEdit();
		QPushButton* btn = new QPushButton(tr("Show zommed axis"));
		QVBoxLayout* l = new QVBoxLayout();
		l->addWidget(_zedit);
		l->addWidget(btn);

		connect(btn, SIGNAL(clicked()), this, SLOT(create_zv()));
		setLayout(l);
	}

protected slots:
	void create_zv()
	{
		PVZoneID zid = _zedit->text().toUInt();
		PVParallelView::PVZoomedParallelView *view = new PVParallelView::PVZoomedParallelView();
		view->setViewport(new QWidget());
		view->setScene(new PVParallelView::PVZoomedParallelScene(view, _zd,
		                                                         zid));
		view->resize(1024, 1024);
		view->show();
	}

private:
	QLineEdit* _zedit;
	PVParallelView::PVZonesDrawing<PARALLELVIEW_ZZT_BBITS>& _zd;
	
	Q_OBJECT
};

#endif

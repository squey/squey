#ifndef ZOOMDLG_H
#define ZOOMDLG_H

#include <pvparallelview/PVZonesDrawing.h>
#include <pvparallelview/PVZoomedParallelScene.h>
#include <pvparallelview/PVZoomedParallelView.h>

#include <picviz/FakePVView.h>

#include <QDialog>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>

class ZoomDlg: public QDialog
{
public:
	ZoomDlg(PVParallelView::PVZonesDrawing<PARALLELVIEW_ZZT_BBITS>& zd,
	        Picviz::FakePVView_p pvview_p,
	        QWidget* parent = NULL):
		_zd(zd),
		_pvview_p(pvview_p)
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
		PVParallelView::PVZoomedParallelView *zpview = new PVParallelView::PVZoomedParallelView();
		zpview->setViewport(new QWidget());
		zpview->setScene(new PVParallelView::PVZoomedParallelScene(zpview,
		                                                           _pvview_p,
		                                                           _zd,
		                                                           zid));
		zpview->resize(1024, 1024);
		zpview->show();
	}

private:
	QLineEdit* _zedit;
	PVParallelView::PVZonesDrawing<PARALLELVIEW_ZZT_BBITS> &_zd;
	Picviz::FakePVView_p                                   &_pvview_p;
	Q_OBJECT
};

#endif

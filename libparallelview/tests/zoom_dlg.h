#ifndef ZOOMDLG_H
#define ZOOMDLG_H

#include <pvparallelview/PVZonesDrawing.h>
#include <pvparallelview/PVZoomedParallelScene.h>
#include <pvparallelview/PVZoomedParallelView.h>
#include <pvparallelview/PVLibView.h>

#include <QDialog>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>

class ZoomDlg: public QDialog
{
	Q_OBJECT

public:
	ZoomDlg(PVParallelView::PVLibView &lv,
	        PVParallelView::PVZonesDrawing<PARALLELVIEW_ZZT_BBITS>& zd,
	        QWidget* parent = nullptr) :
		QDialog(parent),
		_lv(lv),
		_zd(zd)
	{
		_zedit = new QLineEdit();
		QPushButton* btn = new QPushButton(tr("Show zoomed axis"));
		QVBoxLayout* l = new QVBoxLayout();
		l->addWidget(_zedit);
		l->addWidget(btn);

		connect(btn, SIGNAL(clicked()), this, SLOT(create_zv()));
		setLayout(l);
	}

protected slots:
	void create_zv()
	{
		PVCol zid = _zedit->text().toInt();
		PVParallelView::PVZoomedParallelView *zpv = new PVParallelView::PVZoomedParallelView();

		std::cout << "ZoomDlg: zpv: " << zpv << std::endl;
		zpv->setViewport(new QWidget());

		_lv.create_zoomed_scene<PVParallelView::PVBCIDrawingBackendCUDA<PARALLELVIEW_ZZT_BBITS> >(zpv, zid);

		zpv->resize(1024, 1024);
		zpv->show();
	}

private:
	PVParallelView::PVLibView                              &_lv;
	PVParallelView::PVZonesDrawing<PARALLELVIEW_ZZT_BBITS> &_zd;
	QLineEdit                                              *_zedit;
};

#endif

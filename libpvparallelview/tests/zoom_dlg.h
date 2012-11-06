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
	        QWidget* parent = nullptr) :
		QDialog(parent),
		_lv(lv)
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
		PVCol zone_id = _zedit->text().toInt();

		PVParallelView::PVZoomedParallelView *zpv = _lv.create_zoomed_view(zone_id);

		zpv->resize(1024, 1024);
		zpv->show();
	}

private:
	PVParallelView::PVLibView &_lv;
	QLineEdit                 *_zedit;
};

#endif

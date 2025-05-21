//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <QApplication>

#include "mygraphicsview.h"

#include <QApplication>

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QVBoxLayout>
#include <QGroupBox>

#include <QGraphicsItemAnimation>
#include <QGraphicsLineItem>
#include <QTimeLine>

#include <iostream>

#define LINE_COUNT 1024L

#define LINE_NUM LINE_COUNT

#define LINE_STEP (LINE_COUNT / LINE_NUM)

void init_big(QGraphicsScene* scene, PVWidgets::PVGraphicsView*, QTimeLine* timer)
{
	static constexpr auto POS_SCALE = (qreal)(((size_t)1 << 40) * LINE_STEP);
	static constexpr qreal SCENE_WIDTH = ((qreal)LINE_NUM * POS_SCALE);

	for (long i = 0; i < LINE_NUM; ++i) {
		QGraphicsLineItem* item = scene->addLine(0, 0, 0, 100);

		if (i < (LINE_NUM - 5)) {
			item->setPen(QPen(
			    QColor(255 * (i / (double)LINE_NUM), 255 * ((LINE_NUM - i) / (double)LINE_NUM), 0),
			    0));
		}

		auto* animation = new QGraphicsItemAnimation;
		animation->setItem(item);
		animation->setTimeLine(timer);

		animation->setPosAt(0.0, QPointF());
		animation->setPosAt(1.0, QPointF(POS_SCALE * (qreal)i, 0));
		// item->setPos(POS_SCALE * (qreal)i, 0);
	}

	scene->setSceneRect(QRectF(0, 0, SCENE_WIDTH, 100));
	// scene->setSceneRect(QRectF(-SCENE_WIDTH, 0, SCENE_WIDTH, 100));
}

void init_small(QGraphicsScene* scene, PVWidgets::PVGraphicsView*, QTimeLine* timer)
{
	static constexpr auto POS_SCALE = (qreal)((size_t)1 << 63);
	static constexpr qreal SCENE_WIDTH = ((qreal)LINE_NUM * POS_SCALE);

	for (long i = 0; i < LINE_NUM; ++i) {
		QGraphicsLineItem* item =
		    scene->addLine(-i / (qreal)SCENE_WIDTH, 0, -i / (qreal)SCENE_WIDTH, 100);

		if (i < (LINE_NUM - 5)) {
			item->setPen(QPen(
			    QColor(255 * (i / (double)LINE_NUM), 255 * ((LINE_NUM - i) / (double)LINE_NUM), 0),
			    0));
		}
		auto* animation = new QGraphicsItemAnimation;
		animation->setItem(item);
		animation->setTimeLine(timer);

		animation->setPosAt(0.0, QPointF(-i / (qreal)SCENE_WIDTH, 0));
		animation->setPosAt(1.0, QPointF(-i / (qreal)LINE_NUM, 0));
	}

	scene->setSceneRect(QRectF(-1., 0, 1.0, 100));
}

void init_hori_box(QGraphicsScene* scene, PVWidgets::PVGraphicsView*, QGraphicsView*)
{
	scene->addRect(QRectF(-3., -2., 1., 1.));
	scene->addRect(QRectF(-3., 1., 1., 1.));

	scene->addRect(QRectF(2., -2., 1., 1.));
	scene->addRect(QRectF(2., 1., 1., 1.));

	scene->addLine(-1., 0., 1., 0);
	scene->addLine(0., -1., 0., 1.);

	scene->setSceneRect(-3., -2., 6., 4.);
}

using MODE = enum { MODE_BIG = 0, MODE_SMALL = 1, MODE_HBOX = 2 };

int main(int argc, char** argv)
{
	if (argc != 2) {
		std::cerr << "usage: " << argv[0] << " [0|1|2]" << std::endl;
		std::cerr << "     0: test of high order range" << std::endl;
		std::cerr << "     1: test of low order range" << std::endl;
		std::cerr << "     2: test of viewport" << std::endl;
		exit(1);
	}

	QApplication a(argc, argv);

	auto* scene = new QGraphicsScene;

	auto* timer = new QTimeLine(10000);
	timer->setFrameRange(0, 1000);

	auto* pgv = new MyPVGraphicsView(scene);
	QGraphicsView* qgv = new MyQGraphicsView(scene);

	auto* layout = new QVBoxLayout();
	layout->setContentsMargins(0, 0, 0, 0);
	layout->setSpacing(0);

	auto* pgb = new QGroupBox("PV's graphics view");
	auto* qgb = new QGroupBox("Qt's graphics view");

	auto* widget = new QWidget();
	widget->setLayout(layout);

	auto* pgl = new QVBoxLayout();
	pgl->setContentsMargins(0, 0, 0, 0);
	pgl->setSpacing(0);
	pgb->setLayout(pgl);
	pgl->addWidget(pgv, 1);
	layout->addWidget(pgb, 1);

	auto* qgl = new QVBoxLayout();
	qgl->setContentsMargins(0, 0, 0, 0);
	qgl->setSpacing(0);
	qgb->setLayout(qgl);
	qgl->addWidget(qgv, 1);
	layout->addWidget(qgb, 1);

	widget->show();

	int mode = atoi(argv[1]);

	switch (mode) {
	case MODE_BIG:
		init_big(scene, pgv, timer);
		break;
	case MODE_SMALL:
		init_small(scene, pgv, timer);
		break;
	case MODE_HBOX:
		init_hori_box(scene, pgv, qgv);
		break;
	}

	if ((mode == MODE_BIG) || (mode == MODE_SMALL)) {
		QObject::connect(timer, SIGNAL(frameChanged(int)), pgv, SLOT(frame_has_changed(int)));
		QObject::connect(timer, SIGNAL(finished()), pgv, SLOT(anim_finished()));

		QObject::connect(timer, SIGNAL(frameChanged(int)), qgv, SLOT(frame_has_changed(int)));
		QObject::connect(timer, SIGNAL(finished()), qgv, SLOT(anim_finished()));

		widget->resize(3072, 1024);
	} else if (mode == MODE_HBOX) {
		widget->resize(800, 1000);
	}

	pgv->set_scene_rect(scene->sceneRect());
	qgv->setSceneRect(scene->sceneRect());

	timer->start();

	std::cout << "key actions:" << std::endl;
	std::cout << " A: center view ignoring aspect ratio" << std::endl;
	std::cout << " Z: center view using aspect ratio" << std::endl;
	std::cout << " E: center view using aspect ratio by expanding" << std::endl;

	std::cout << " Q: set scrollbars policy to AsNeeded" << std::endl;
	std::cout << " S: set scrollbars policy to AlwaysOff" << std::endl;
	std::cout << " D: set scrollbars policy to AlwaysOn" << std::endl;

	std::cout << " W: center on last clicked mouse position" << std::endl;
	std::cout << " X: set transform to scale (1., 1.)" << std::endl;

	return a.exec();

	return 0;
}

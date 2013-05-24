#include <QApplication>
#include <QGraphicsScene>
//#include <QGraphicsView>
#include <QGraphicsLineItem>
#include <QDebug>
#include <QWheelEvent>

#include <pvkernel/widgets/PVGraphicsView.h>

#define HEIGHT 1024
#define STEP 16

class EventFilter: public QObject
{
protected:
	bool eventFilter(QObject* obj, QEvent* ev)
	{
		if (ev->type() == QEvent::Wheel) {
			PVWidgets::PVGraphicsView* view = static_cast<PVWidgets::PVGraphicsView*>(obj);
			QWheelEvent* wev = static_cast<QWheelEvent*>(ev);
			if (wev->modifiers() == Qt::AltModifier) {
				if (wev->delta() > 0) {
					scale *= 1.2;
				}
				else {
					scale /= 1.2;
				}
				QTransform tscale;
				tscale.scale(1.0, scale);
				view->set_transform(tscale * inv_y);
				view->get_viewport()->update();
				return true;
			}
		}
		return QObject::eventFilter(obj, ev);
	}

public:
	QTransform inv_y;
	double scale;
};

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	size_t height = HEIGHT;
	size_t step = STEP;
	if (argc > 2) {
		height = atoll(argv[1]);
		step = atoll(argv[2]);
	}

	PVWidgets::PVGraphicsView* view = new PVWidgets::PVGraphicsView(new QGraphicsScene());
	QGraphicsScene* scene = view->get_scene();
	//QGraphicsScene* scene = new QGraphicsScene();
	//QGraphicsView* view = new QGraphicsView();

	scene->setItemIndexMethod(QGraphicsScene::NoIndex);
	for (size_t y = 0; y < height; y += step) {
		QGraphicsLineItem* item = new QGraphicsLineItem(0, y, 1024, y);
		const double r = (double)y/(double)height;
		item->setPen(QColor(r * 255.0, (1.0-r)*255.0, (1.0-r)*255.0));
		scene->addItem(item);
	}

	//view->setScene(scene);

	qDebug() << scene->sceneRect();

	QTransform inv_y;
	inv_y.translate(0.0, scene->sceneRect().height());
	inv_y.scale(1.0, -1.0);

	EventFilter* evf = new EventFilter();
	evf->scale = 8192.0/(double)height;
	evf->inv_y = inv_y;

	QTransform scale;
	scale.scale(1.0, evf->scale);

	qDebug() << view->get_transform();
	view->set_transform(scale * inv_y, true);
	qDebug() << view->get_transform();

	view->installEventFilter(evf);

	view->show();

	return app.exec();
}

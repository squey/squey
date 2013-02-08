
#ifndef MYGRAPHICSVIEW_H
#define MYGRAPHICSVIEW_H

#include <pvkernel/widgets/PVGraphicsView.h>

#include <QGraphicsView>
#include <QKeyEvent>

#include <iostream>


#define print_rect(R) __print_rect(#R, R)

template <typename R>
void __print_rect(const char *text, const R &r)
{
	std::cout << text << ": "
	          << r.x() << " " << r.y() << ", "
	          << r.width() << " " << r.height()
	          << std::endl;
}

#define print_transform(T) __print_transform(#T, T)

template <typename T>
void __print_transform(const char *text, const T &t)
{
	std::cout << text << ": " << std::endl
	          << t.m11() << " " << t.m21() << " " << t.m31() << std::endl
	          << t.m12() << " " << t.m22() << " " << t.m32() << std::endl
	          << t.m13() << " " << t.m23() << " " << t.m33() << std::endl;
}

class MyPVGraphicsView : public PVWidgets::PVGraphicsView
{
	Q_OBJECT

	public:
	MyPVGraphicsView(QGraphicsScene *scene = nullptr,
	                 QWidget *parent = nullptr) :
		PVGraphicsView(scene, parent)
	{}

	void keyPressEvent(QKeyEvent *event)
	{
		event->accept();
		if (event->key() == Qt::Key_A) {
			fit_in_view(Qt::IgnoreAspectRatio);
		} else if (event->key() == Qt::Key_Z) {
			fit_in_view(Qt::KeepAspectRatio);
		} else if (event->key() == Qt::Key_E) {
			fit_in_view(Qt::KeepAspectRatioByExpanding);
		} else if (event->key() == Qt::Key_Q) {
			set_horizontal_scrollbar_policy(Qt::ScrollBarAsNeeded);
			set_vertical_scrollbar_policy(Qt::ScrollBarAsNeeded);
		} else if (event->key() == Qt::Key_S) {
			set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOff);
			set_vertical_scrollbar_policy(Qt::ScrollBarAlwaysOff);
		} else if (event->key() == Qt::Key_D) {
			set_horizontal_scrollbar_policy(Qt::ScrollBarAlwaysOn);
			set_vertical_scrollbar_policy(Qt::ScrollBarAlwaysOn);
		} else if (event->key() == Qt::Key_W) {
			center_on(map_to_scene(_mouse_pos));
		} else if (event->key() == Qt::Key_X) {
			QTransform t;
			t.scale(1., 1.);
			set_transform(t);;
		}
		update();
	}

	void mousePressEvent(QMouseEvent *event)
	{
		_mouse_pos = event->pos();
	}

	virtual void drawBackground(QPainter *painter, const QRectF &rect)
	{
		// painter->fillRect(rect, QColor(255, 200, 200));
		painter->fillRect(rect, Qt::white);
	}

	virtual void drawForeground(QPainter *, const QRectF &)
	{}

public slots:
	void frame_has_changed (int frame)
	{
		if (get_scene() == nullptr) {
			return;
		}

		update();
	}

	void anim_finished()
	{
		std::cout << "MyPVGraphicsView::animation stops" << std::endl;
	}

private:
	QPointF _mouse_pos;
};

class MyQGraphicsView : public QGraphicsView
{
	Q_OBJECT
	public:
	MyQGraphicsView(QGraphicsScene *scene = nullptr, QWidget *parent = nullptr) :
		QGraphicsView(scene, parent)
	{}

	void keyPressEvent(QKeyEvent *event)
	{
		event->accept();
		if (event->key() == Qt::Key_A) {
			fitInView(scene()->sceneRect(), Qt::IgnoreAspectRatio);
		} else if (event->key() == Qt::Key_Z) {
			fitInView(scene()->sceneRect(), Qt::KeepAspectRatio);
		} else if (event->key() == Qt::Key_E) {
			fitInView(scene()->sceneRect(), Qt::KeepAspectRatioByExpanding);
		} else if (event->key() == Qt::Key_Q) {
			setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
			setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
		} else if (event->key() == Qt::Key_S) {
			setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
			setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
		} else if (event->key() == Qt::Key_D) {
			setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
			setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
		} else if (event->key() == Qt::Key_W) {
			centerOn(mapToScene(_mouse_pos));
		} else if (event->key() == Qt::Key_X) {
			QTransform t;
			t.scale(1., 1.);
			setTransform(t);;
		}
		update();
	}

	void mousePressEvent(QMouseEvent *event)
	{
		_mouse_pos = event->pos();
		QGraphicsView::mousePressEvent(event);
	}

	void paintEvent(QPaintEvent *event)
	{
		// QRect view_area = event->rect().intersected(viewport()->rect());
		// QRectF scene_area = mapToScene(view_area).boundingRect();
		// std::cout << "##########################################################" << std::endl;
		// std::cout << "MyQGraphicsView::paintEvent()" << std::endl;
		// print_rect(view_area);
		// print_rect(scene_area);
		// print_rect(sceneRect());
		// print_rect(scene()->sceneRect());

		QGraphicsView::paintEvent(event);
	}

public slots:
	void frame_has_changed (int frame)
	{
		if (scene() == nullptr) {
			return;
		}

		update();
	}

	void anim_finished()
	{
		std::cout << "MyQGraphicsView::animation stops" << std::endl;
	}

private:
	QPoint _mouse_pos;
};

#endif // MYGRAPHICSVIEW_H

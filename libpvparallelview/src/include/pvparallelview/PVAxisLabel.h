/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVAXISLABEL_H
#define PVPARALLELVIEW_PVAXISLABEL_H

#include <QObject>
#include <QGraphicsSimpleTextItem>
#include <QGraphicsSceneContextMenuEvent>
#include <QBrush>
#include <QPainterPath>

namespace Inendi
{

class PVView;
}

namespace PVParallelView
{

class PVAxisGraphicsItem;

class PVAxisLabel : public QObject, public QGraphicsSimpleTextItem
{
	Q_OBJECT
  private:
	static constexpr int MAX_WIDTH =
	    120; /*!< The maximum width of a label in pixel. This value should be calculated later,
	           depend of the client's windows settings. */

  public:
	explicit PVAxisLabel(const Inendi::PVView& view, QGraphicsItem* parent = nullptr);

	virtual ~PVAxisLabel();

	/** Elide the text if it is longer than MAX_WIDTH.*/
	void set_text(const QString& text);

	void set_color(const QColor& color) { setBrush(color); }

	QRectF get_scene_bbox() { return mapRectToScene(boundingRect()); }

	void set_bounding_box_width(int width);
	bool contains(const QPointF& point) const override;
	QPainterPath shape() const override;
	QRectF boundingRect() const override;

  private:
	PVAxisGraphicsItem const* get_parent_axis() const;

  private:
	const Inendi::PVView& _lib_view;
	int _bounding_box_width = 0;
};
}

#endif // PVPARALLELVIEW_PVAXISLABEL_H

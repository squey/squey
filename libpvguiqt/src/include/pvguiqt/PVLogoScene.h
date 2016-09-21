/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef OPENGLSCENE_H
#define OPENGLSCENE_H

#include "pvguiqt/PVPoint3D.h"

#include <QGraphicsScene>
#include <QLabel>
#include <QTime>

namespace PVGuiQt
{

class PVLogoModel;

class PVLogoScene : public QGraphicsScene
{
	Q_OBJECT

  public:
	PVLogoScene();

	void drawBackground(QPainter* painter, const QRectF& rect) override;

  public Q_SLOTS:
	void enableWireframe(bool enabled);
	void enableNormals(bool enabled);
	void setModelColor();
	void loadModel(const QString& filePath);

  protected:
	void mousePressEvent(QGraphicsSceneMouseEvent* event) override;
	void mouseReleaseEvent(QGraphicsSceneMouseEvent* event) override;
	void mouseMoveEvent(QGraphicsSceneMouseEvent* event) override;

  private:
	QDialog* createDialog(const QString& windowTitle) const;

	void setModel(PVLogoModel* model);

	bool m_wireframeEnabled;
	bool m_normalsEnabled;

	QColor m_modelColor;

	PVLogoModel* m_model;

	QTime m_time;
	int m_lastTime;
	int m_mouseEventTime;

	float m_distance;
	PVPoint3D m_rotation;
	PVPoint3D m_angularMomentum;
	PVPoint3D m_accumulatedMomentum;

	PVPoint3D m_lightPosition;
};
} // namespace PVGuiQt

#endif

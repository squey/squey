#ifndef OPENGLSCENE_H
#define OPENGLSCENE_H

#include "pvguiqt/PVPoint3D.h"

#include <QGraphicsScene>
#include <QLabel>
#include <QTime>

#ifndef QT_NO_CONCURRENT
#include <QFutureWatcher>
#endif

namespace PVGuiQt
{

class PVLogoModel;

class PVLogoScene : public QGraphicsScene
{
    Q_OBJECT

public:
    PVLogoScene();

    void drawBackground(QPainter* painter, const QRectF &rect);

public slots:
    void enableWireframe(bool enabled);
    void enableNormals(bool enabled);
    void setModelColor();
    void loadModel(const QString &filePath);
    void modelLoaded();

protected:
    void mousePressEvent(QGraphicsSceneMouseEvent* event);
    void mouseReleaseEvent(QGraphicsSceneMouseEvent* event);
    void mouseMoveEvent(QGraphicsSceneMouseEvent* event);

private:
    QDialog* createDialog(const QString &windowTitle) const;

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

    QGraphicsRectItem* m_lightItem;

#ifndef QT_NO_CONCURRENT
    QFutureWatcher<PVLogoModel*> m_modelLoader;
#endif
};

}

#endif

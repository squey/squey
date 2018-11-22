/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2018
 */

#ifndef _PVSERIESVIEW_H_
#define _PVSERIESVIEW_H_

#include <inendi/PVRangeSubSampler.h>

#include <QOpenGLVertexArrayObject>
#include <QOpenGLFunctions>
#include <QtGlobal>
#if QT_VERSION < QT_VERSION_CHECK(5, 4, 0)
#include <QGLWidget>
#include <QGLShaderProgram>
#include <QGLShader>
#include <QGLBuffer>
using PVOpenGLWidget = QGLWidget;
using QOpenGLShader = QGLShader;
using QOpenGLShaderProgram = QGLShaderProgram;
using QOpenGLBuffer = QGLBuffer;
#else
#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLShader>
#include <QOpenGLBuffer>
using PVOpenGLWidget = QOpenGLWidget;
#endif

class PVSeriesView : public PVOpenGLWidget, protected QOpenGLFunctions
{
	Q_OBJECT

  public:
	explicit PVSeriesView(Inendi::PVRangeSubSampler& rss, QWidget* parent = 0);
	virtual ~PVSeriesView();

	void setBackgroundColor(QColor const& bgcol);

  protected:
  public:
	void initializeGL() override;
	void resizeGL(int w, int h) override;
	void paintGL() override;

	void debugAvailableMemory();

  private:
	struct Vertex {
		// GLfloat x;
		// GLfloat y;
		GLushort y;
	};

	Inendi::PVRangeSubSampler& m_rss;

	QOpenGLVertexArrayObject m_vao;
	QOpenGLBuffer m_vbo;
	QOpenGLBuffer m_dbo;
	std::unique_ptr<QOpenGLShaderProgram> m_program;

	std::optional<QColor> m_backgroundColor;

	int m_verticesCount = 0;
	int m_linesPerVboCount = 0;
	size_t m_linesCount = 0;

	int m_batches = 1;

	int m_sizeLocation = 0;

	int m_w = 0, m_h = 0;

  public:
	QRectF m_vpRect;

	void (*glMultiDrawArraysIndirect)(GLenum mode,
	                                  const void* indirect,
	                                  GLsizei drawcount,
	                                  GLsizei stride) = nullptr;
	void* (*glMapBufferRange)(GLenum target,
	                          GLintptr offset,
	                          GLsizeiptr length,
	                          GLbitfield access) = nullptr;
};

#include <QGraphicsView>
#include <QResizeEvent>

class PVDecoratedSeriesView : public QGraphicsView
{
	Q_OBJECT

  public:
	PVDecoratedSeriesView(QGraphicsScene* scene) : QGraphicsView(scene) {}
	virtual ~PVDecoratedSeriesView() = default;

	void setupViewport(QWidget* widget) override
	{
		QGraphicsView::setupViewport(widget);
		static_cast<PVSeriesView*>(viewport())->makeCurrent();
		static_cast<PVSeriesView*>(viewport())->initializeGL();
		static_cast<PVSeriesView*>(viewport())->doneCurrent();
	}

	void resizeEvent(QResizeEvent* event) override
	{
		QGraphicsView::resizeEvent(event);
		qDebug() << "resizeEvent" << event->size();

		static_cast<PVSeriesView*>(viewport())->makeCurrent();
		static_cast<PVSeriesView*>(viewport())
		    ->resizeGL(event->size().width(), event->size().height());
		static_cast<PVSeriesView*>(viewport())->doneCurrent();
	}

	void paintEvent(QPaintEvent* event) override
	{
		qDebug() << "paintEvent";
#if 0

	    Q_D(QGraphicsView);
	    if (!d->scene) {
	        QAbstractScrollArea::paintEvent(event);
	        return;
	    }

	    // Set up painter state protection.
	    d->scene->d_func()->painterStateProtection = !(d->optimizationFlags & DontSavePainterState);

	    // Determine the exposed region
	    d->exposedRegion = event->region();
	    QRectF exposedSceneRect = mapToScene(d->exposedRegion.boundingRect()).boundingRect();

	    // Set up the painter
	    QPainter painter(viewport());
	    // Set up render hints
	    painter.setRenderHints(painter.renderHints(), false);
	    painter.setRenderHints(d->renderHints, true);

	    // Set up viewport transform
	    const bool viewTransformed = isTransformed();
	    if (viewTransformed)
	        painter.setWorldTransform(viewportTransform());
	    const QTransform viewTransform = painter.worldTransform();

	    // Draw background
	    {
	        if (!(d->optimizationFlags & DontSavePainterState))
	            painter.save();
	        drawBackground(&painter, exposedSceneRect);
	        if (!(d->optimizationFlags & DontSavePainterState))
	            painter.restore();
	    }

	    // Items
	    if (!(d->optimizationFlags & IndirectPainting)) {
	        const quint32 oldRectAdjust = d->scene->d_func()->rectAdjust;
	        if (d->optimizationFlags & QGraphicsView::DontAdjustForAntialiasing)
	            d->scene->d_func()->rectAdjust = 1;
	        else
	            d->scene->d_func()->rectAdjust = 2;
	        d->scene->d_func()->drawItems(&painter, viewTransformed ? &viewTransform : 0,
	                                      &d->exposedRegion, viewport());
	        d->scene->d_func()->rectAdjust = oldRectAdjust;
	        // Make sure the painter's world transform is restored correctly when
	        // drawing without painter state protection (DontSavePainterState).
	        // We only change the worldTransform() so there's no need to do a full-blown
	        // save() and restore(). Also note that we don't have to do this in case of
	        // IndirectPainting (the else branch), because in that case we always save()
	        // and restore() in QGraphicsScene::drawItems().
	        if (!d->scene->d_func()->painterStateProtection)
	            painter.setOpacity(1.0);
	        painter.setWorldTransform(viewTransform);
	    } else {
	        // Make sure we don't have unpolished items before we draw
	        if (!d->scene->d_func()->unpolishedItems.isEmpty())
	            d->scene->d_func()->_q_polishItems();
	        // We reset updateAll here (after we've issued polish events)
	        // so that we can discard update requests coming from polishEvent().
	        d->scene->d_func()->updateAll = false;

	        // Find all exposed items
	        bool allItems = false;
	        QList<QGraphicsItem *> itemList = d->findItems(d->exposedRegion, &allItems, viewTransform);
	        if (!itemList.isEmpty()) {
	            // Generate the style options.
	            const int numItems = itemList.size();
	            QGraphicsItem **itemArray = &itemList[0]; // Relies on QList internals, but is perfectly valid.
	            QStyleOptionGraphicsItem *styleOptionArray = d->allocStyleOptionsArray(numItems);
	            QTransform transform(Qt::Uninitialized);
	            for (int i = 0; i < numItems; ++i) {
	                QGraphicsItem *item = itemArray[i];
	                QGraphicsItemPrivate *itemd = item->d_ptr.data();
	                itemd->initStyleOption(&styleOptionArray[i], viewTransform, d->exposedRegion, allItems);
	                // Cache the item's area in view coordinates.
	                // Note that we have to do this here in case the base class implementation
	                // (QGraphicsScene::drawItems) is not called. If it is, we'll do this
	                // operation twice, but that's the price one has to pay for using indirect
	                // painting :-/.
	                const QRectF brect = adjustedItemEffectiveBoundingRect(item);
	                if (!itemd->itemIsUntransformable()) {
	                    transform = item->sceneTransform();
	                    if (viewTransformed)
	                        transform *= viewTransform;
	                } else {
	                    transform = item->deviceTransform(viewTransform);
	                }
	                itemd->paintedViewBoundingRects.insert(d->viewport, transform.mapRect(brect).toRect());
	            }
	            // Draw the items.
	            drawItems(&painter, numItems, itemArray, styleOptionArray);
	            d->freeStyleOptionsArray(styleOptionArray);
	        }
	    }

	    // Foreground
	    drawForeground(&painter, exposedSceneRect);

	    painter.end();

	    // Restore painter state protection.
	    d->scene->d_func()->painterStateProtection = true;
#endif
		QGraphicsView::paintEvent(event);
	}

	void drawBackground(QPainter* painter, QRectF const& rect) override
	{
		qDebug() << "drawBackground" << rect;
		QGraphicsView::drawBackground(painter, rect);
		painter->beginNativePainting();
		// static_cast<PVSeriesView*>(viewport())->m_vpRect = rect;
		// static_cast<PVSeriesView*>(viewport())->updateGL();

		// static_cast<PVSeriesView*>(viewport())->makeCurrent();
		static_cast<PVSeriesView*>(viewport())->paintGL();
		// static_cast<PVSeriesView*>(viewport())->doneCurrent();
		painter->endNativePainting();
	}

	void drawForeground(QPainter* painter, QRectF const& rect) override
	{
		painter->beginNativePainting();
		// static_cast<PVSeriesView*>(viewport())->m_vpRect = rect;
		// static_cast<PVSeriesView*>(viewport())->updateGL();

		// static_cast<PVSeriesView*>(viewport())->makeCurrent();
		static_cast<PVSeriesView*>(viewport())->paintGL();
		// static_cast<PVSeriesView*>(viewport())->doneCurrent();
		painter->endNativePainting();
	}
};

#endif // _PVSERIESVIEW_H_
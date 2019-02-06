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
#include <QOpenGLTexture>
#include <QOpenGLFramebufferObject>
#include <QOpenGLWidget>
#include <QOpenGLExtraFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLShader>
#include <QOpenGLBuffer>
#include <QBasicTimer>

namespace PVParallelView
{

class PVSeriesAbstractRenderer;

class PVSeriesView : public QWidget
{
	Q_OBJECT

  public:
	struct SerieDrawInfo {
		size_t dataIndex;
		QColor color;
	};

	explicit PVSeriesView(Inendi::PVRangeSubSampler& rss, QWidget* parent = 0);
	virtual ~PVSeriesView();

	void setBackgroundColor(QColor const& bgcol);
	void showSeries(std::vector<SerieDrawInfo> seriesDrawOrder);
	void onResampled();

  protected:
	void paintEvent(QPaintEvent* event) override;
	void resizeEvent(QResizeEvent* event) override;
	void timerEvent(QTimerEvent* event) override;

  private:
	std::unique_ptr<PVSeriesAbstractRenderer> m_renderer;
	QPixmap m_pixmap;
	bool m_needHardRedraw = false;
	QBasicTimer m_resizingTimer;

	Inendi::PVRangeSubSampler& m_rss;
};

} // namespace PVParallelView

#endif // _PVSERIESVIEW_H_
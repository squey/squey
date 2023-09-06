/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __PVGUIQT_PVABOUTBOXDIALOG_H__
#define __PVGUIQT_PVABOUTBOXDIALOG_H__

#include <QDialog>
#include <QHBoxLayout>
#include <QGraphicsView>
#include <QKeyEvent>
#include <QResizeEvent>
#include <QWidget>

// #include <Qt3DCore/QEntity>
// #include <Qt3DRender/QCamera>
// #include <Qt3DRender/QMesh>
// #include <Qt3DRender/QCameraLens>
// #include <Qt3DCore/QTransform>
// #include <Qt3DCore/QAspectEngine>

// #include <Qt3DInput/QInputAspect>

// #include <Qt3DRender/QRenderAspect>
// #include <Qt3DExtras/QForwardRenderer>
// #include <Qt3DExtras/QPhongMaterial>
// #include <QPropertyAnimation>
// #include <Qt3DExtras/Qt3DWindow>

class QTabWidget;

namespace PVGuiQt
{

class PVAboutBoxDialog : public QDialog
{
  public:
	enum Tab { SOFTWARE, CHANGELOG, REFERENCE_MANUAL, OPEN_SOURCE_SOFTWARE };

  public:
	explicit PVAboutBoxDialog(Tab = SOFTWARE, QWidget* parent = nullptr, QVariant data = {});
	void select_tab(Tab);

  private:
	QHBoxLayout* _view3D_layout;

	QTabWidget* _tab_widget;
	QWidget* _changelog_tab;
};

namespace __impl
{

// class OrbitTransformController : public QObject
// {
// 	Q_OBJECT
// 	Q_PROPERTY(Qt3DCore::QTransform* target READ target WRITE setTarget NOTIFY targetChanged)
// 	Q_PROPERTY(float radius READ radius WRITE setRadius NOTIFY radiusChanged)
// 	Q_PROPERTY(float angle READ angle WRITE setAngle NOTIFY angleChanged)

//   public:
// 	OrbitTransformController(QObject* parent = 0);

// 	void setTarget(Qt3DCore::QTransform* target);
// 	Qt3DCore::QTransform* target() const;

// 	void setRadius(float radius);
// 	float radius() const;

// 	void setAngle(float angle);
// 	float angle() const;

//   Q_SIGNALS:
// 	void targetChanged();
// 	void radiusChanged();
// 	void angleChanged();

//   protected:
// 	void updateMatrix();

//   private:
// 	Qt3DCore::QTransform* m_target;
// 	QMatrix4x4 m_matrix;
// 	float m_radius;
// 	float m_angle;
// };

} // namespace __impl
} // namespace PVGuiQt

#endif /* __PVGUIQT_PVABOUTBOXDIALOG_H__ */

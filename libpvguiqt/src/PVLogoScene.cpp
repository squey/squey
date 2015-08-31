#include "pvguiqt/PVLogoScene.h"
#include "pvguiqt/PVLogoModel.h"

#include <QtWidgets>
#include <QtOpenGL>

#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE  0x809D
#endif

// WORKAROUND
void gluPerspective(double fovy, double aspect, double zNear, double zFar)
{
	// Start in projection mode.
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	double xmin, xmax, ymin, ymax;
	ymax = zNear * tan(fovy * M_PI / 360.0);
	ymin = -ymax;
	xmin = ymin * aspect;
	xmax = ymax * aspect;
	glFrustum(xmin, xmax, ymin, ymax, zNear, zFar);
}

QDialog* PVGuiQt::PVLogoScene::createDialog(const QString &windowTitle) const
{
    QDialog* dialog = new QDialog(0, Qt::CustomizeWindowHint | Qt::WindowTitleHint);

    dialog->setWindowOpacity(0.8);
    dialog->setWindowTitle(windowTitle);
    dialog->setLayout(new QVBoxLayout);

    return dialog;
}

PVGuiQt::PVLogoScene::PVLogoScene()
    : m_wireframeEnabled(false)
    , m_normalsEnabled(false)
    , m_modelColor(0x78, 0x1b, 0x7d)
    , m_model(0)
    , m_lastTime(0)
    , m_distance(1.15f)
    , m_angularMomentum(0, 100, 0)
	, m_lightPosition(0, 0, 512)
{

#ifndef QT_NO_CONCURRENT
	connect(&m_modelLoader, SIGNAL(finished()), this, SLOT(modelLoaded()));
#endif

    loadModel(QLatin1String(":/logo3d"));
    m_time.start();
}

void PVGuiQt::PVLogoScene::drawBackground(QPainter *painter, const QRectF &)
{
    if (painter->paintEngine()->type() != QPaintEngine::OpenGL
        && painter->paintEngine()->type() != QPaintEngine::OpenGL2)
    {
        qWarning("OpenGLScene: drawBackground needs a QOpenGLWidget to be set as viewport on the graphics view");
        return;
    }

    painter->beginNativePainting();

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (m_model) {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        gluPerspective(70, width() / height(), 0.01, 1000);

        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();

        //const float pos[] = { (m_lightPosition.x - width() / 2), (height() / 2 - m_lightPosition.y), m_lightPosition.z, 0 };
        //glLightfv(GL_LIGHT0, GL_POSITION, pos);
        glColor4f(m_modelColor.redF(), m_modelColor.greenF(), m_modelColor.blueF(), 1.0f);

        const int delta = m_time.elapsed() - m_lastTime;
        m_rotation += m_angularMomentum * (delta / 1000.0);
        m_lastTime += delta;

        glTranslatef(0, 0, -m_distance);
        glRotatef(m_rotation.x, 1, 0, 0);
        glRotatef(m_rotation.y, 0, 1, 0);
        glRotatef(m_rotation.z, 0, 0, 1);

        glEnable(GL_MULTISAMPLE);
        m_model->render(m_wireframeEnabled, m_normalsEnabled);
        glDisable(GL_MULTISAMPLE);

        glPopMatrix();

        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
    }

    painter->endNativePainting();

    QTimer::singleShot(20, this, SLOT(update()));
}

/*static PVGuiQt::PVLogoModel* PVGuiQt::PVLogoScene::loadModel(const QString &filePath)
{
    return (new PVGuiQt::PVLogoModel(filePath)
}*/

void PVGuiQt::PVLogoScene::loadModel(const QString &filePath)
{
    if (filePath.isEmpty())
        return;

    QApplication::setOverrideCursor(Qt::BusyCursor);
    setModel(new PVGuiQt::PVLogoModel(filePath)); //  setModel(::loadModel(filePath));
    modelLoaded();
    QApplication::restoreOverrideCursor();
}

void PVGuiQt::PVLogoScene::modelLoaded()
{
}

void PVGuiQt::PVLogoScene::setModel(PVLogoModel *model)
{
    delete m_model;
    m_model = model;

    update();
}

void PVGuiQt::PVLogoScene::enableWireframe(bool enabled)
{
    m_wireframeEnabled = enabled;
    update();
}

void PVGuiQt::PVLogoScene::enableNormals(bool enabled)
{
    m_normalsEnabled = enabled;
    update();
}

void PVGuiQt::PVLogoScene::setModelColor()
{
    const QColor color = QColorDialog::getColor(m_modelColor);
    if (color.isValid()) {
        m_modelColor = color;
        update();
    }
}

void PVGuiQt::PVLogoScene::mouseMoveEvent(QGraphicsSceneMouseEvent *event)
{
    QGraphicsScene::mouseMoveEvent(event);
    if (event->isAccepted())
        return;
    if (event->buttons() & Qt::LeftButton) {
        const QPointF delta = event->scenePos() - event->lastScenePos();
        const PVPoint3D angularImpulse = PVPoint3D(delta.y(), delta.x(), 0) * 0.1;

        m_rotation += angularImpulse;
        m_accumulatedMomentum += angularImpulse;

        event->accept();
        update();
    }
}

void PVGuiQt::PVLogoScene::mousePressEvent(QGraphicsSceneMouseEvent *event)
{
    QGraphicsScene::mousePressEvent(event);
    if (event->isAccepted())
        return;

    m_mouseEventTime = m_time.elapsed();
    m_angularMomentum = m_accumulatedMomentum = PVPoint3D();
    event->accept();
}

void PVGuiQt::PVLogoScene::mouseReleaseEvent(QGraphicsSceneMouseEvent *event)
{
    QGraphicsScene::mouseReleaseEvent(event);
    if (event->isAccepted())
        return;

    const int delta = m_time.elapsed() - m_mouseEventTime;
    m_angularMomentum = m_accumulatedMomentum * (1000.0 / qMax(1, delta));
    event->accept();
    update();
}

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

#include <pvbase/general.h>

#include <pvguiqt/PVAboutBoxDialog.h>

#include <iostream>
#include <iterator>
#include <sstream>

#include <QApplication>
#include <QLabel>
#include <QGridLayout>
#include <QPushButton>
#include <QTabWidget>
#include <QListWidget>
#include <QDirIterator>
#include <QTextEdit>
#include <QTextStream>
#include <QtWebEngineWidgets/QWebEngineView>
#include <QDebug>

#include <pvparallelview/PVSeriesRendererOffscreen.h>
#include <pvkernel/opencl/common.h>

#include <cassert>

static QString copying_dir()
{
	const char* path = getenv("COPYING_DIR");
	if (path) {
		return path;
	}
	return INENDI_COPYING_DIR;
}

class PVOpenSourceSoftwareWidget : public QWidget
{
  public:
	PVOpenSourceSoftwareWidget(QWidget* parent = nullptr) : QWidget(parent)
	{
		QListWidget* oss_software_list = new QListWidget;

		QDirIterator dir_it(copying_dir(), QDir::Files);
		while (dir_it.hasNext()) {
			oss_software_list->addItem(QFileInfo(dir_it.next()).fileName());
		}
		oss_software_list->sortItems();
		oss_software_list->setMaximumWidth(oss_software_list->sizeHintForColumn(0) + 2);

		QTextEdit* license_text = new QTextEdit;
		license_text->setReadOnly(true);

		QHBoxLayout* layout = new QHBoxLayout;

		layout->addWidget(oss_software_list);
		layout->addWidget(license_text);

		connect(oss_software_list, &QListWidget::currentRowChanged, [=]() {
			const QString& file_path =
			    copying_dir() + "/" + oss_software_list->currentItem()->text();
			QFile f(file_path);
			f.open(QFile::ReadOnly | QFile::Text);
			QTextStream in(&f);
			license_text->setText(in.readAll());
		});

		oss_software_list->setCurrentRow(0);

		setLayout(layout);
	}
};

class PVChangeLogWidget : public QWidget
{
  public:
	PVChangeLogWidget(QWidget* parent = nullptr) : QWidget(parent)
	{
		QTextEdit* changelog_text = new QTextEdit;
		changelog_text->setReadOnly(true);

		QHBoxLayout* layout = new QHBoxLayout;

		layout->addWidget(changelog_text);

		QFile f("/app/share/inendi/inspector/CHANGELOG");
		f.open(QFile::ReadOnly | QFile::Text);
		QTextStream in(&f);
		changelog_text->setText(in.readAll());

		setLayout(layout);
	}
};

PVGuiQt::__impl::OrbitTransformController::OrbitTransformController(QObject* parent)
    : QObject(parent), m_target(nullptr), m_matrix(), m_radius(1.0f), m_angle(0.0f)
{
}

void PVGuiQt::__impl::OrbitTransformController::setTarget(Qt3DCore::QTransform* target)
{
	if (m_target != target) {
		m_target = target;
		targetChanged();
	}
}

Qt3DCore::QTransform* PVGuiQt::__impl::OrbitTransformController::target() const
{
	return m_target;
}

void PVGuiQt::__impl::OrbitTransformController::setRadius(float radius)
{
	if (!qFuzzyCompare(radius, m_radius)) {
		m_radius = radius;
		updateMatrix();
		radiusChanged();
	}
}

float PVGuiQt::__impl::OrbitTransformController::radius() const
{
	return m_radius;
}

void PVGuiQt::__impl::OrbitTransformController::setAngle(float angle)
{
	if (!qFuzzyCompare(angle, m_angle)) {
		m_angle = angle;
		updateMatrix();
		angleChanged();
	}
}

float PVGuiQt::__impl::OrbitTransformController::angle() const
{
	return m_angle;
}

void PVGuiQt::__impl::OrbitTransformController::updateMatrix()
{
	m_matrix.setToIdentity();
	m_matrix.rotate(m_angle, QVector3D(0.0f, 1.0f, 0.0f));
	m_matrix.translate(m_radius, 0.0f, 0.0f);
	m_target->setMatrix(m_matrix);
}

Qt3DCore::QEntity* createScene()
{
	// Root entity
	Qt3DCore::QEntity* rootEntity = new Qt3DCore::QEntity;

	// Material
	auto* material = new Qt3DExtras::QPhongMaterial(rootEntity);
	material->setAmbient(QColor(0xf1, 0x40, 0x00, 0xff));
	//material->setDiffuse(QColor(0xf1, 0x59, 0x22, 0xff));

	Qt3DCore::QTransform* meshTransform = new Qt3DCore::QTransform;
	meshTransform->setScale3D(QVector3D(1, 1, 1));
	meshTransform->setRotation(QQuaternion::fromAxisAndAngle(QVector3D(1, 0, 0), 0.0f));
	meshTransform->setTranslation({0,-100,0});

	auto* controller = new PVGuiQt::__impl::OrbitTransformController(meshTransform);
	controller->setTarget(meshTransform);
	controller->setRadius(0.0f);

	QPropertyAnimation* meshRotateTransformAnimation = new QPropertyAnimation(meshTransform);
	meshRotateTransformAnimation->setTargetObject(controller);
	meshRotateTransformAnimation->setPropertyName("angle");
	meshRotateTransformAnimation->setStartValue(QVariant::fromValue(0));
	meshRotateTransformAnimation->setEndValue(QVariant::fromValue(360));
	meshRotateTransformAnimation->setDuration(4000);
	meshRotateTransformAnimation->setLoopCount(-1);
	meshRotateTransformAnimation->start();

	Qt3DCore::QEntity* meshEntity = new Qt3DCore::QEntity(rootEntity);
	auto mesh = new Qt3DRender::QMesh();
	QUrl data = QUrl::fromLocalFile(":/logo3d");
	mesh->setSource(data);
	meshEntity->addComponent(mesh);
	meshEntity->addComponent(meshTransform);
	meshEntity->addComponent(material);

	return rootEntity;
}

PVGuiQt::PVAboutBoxDialog::PVAboutBoxDialog(Tab tab /*= SOFTWARE*/, QWidget* parent /*= 0*/, QVariant /* data  = {} */)
    : QDialog(parent)
{
	setWindowTitle("About INENDI Inspector");

	auto main_layout = new QGridLayout;
	main_layout->setHorizontalSpacing(0);

	QString content = "INENDI Inspector version \"" + QString(INENDI_CURRENT_VERSION_STR) + "\"";

	content += "<br/>website - <a "
	           "href=\"https://gitlab.com/inendi/inspector\">gitlab.com/inendi/inspector</a><br/>";
	content += QString("documentation") + " - <a href=\"" + DOC_URL +"\">" + QString(DOC_URL).replace("https://","") + "</a><br/>";
	content += "contact - <a href=\"mailto:";
	content += EMAIL_ADDRESS_CONTACT;
	content += "?subject=%5BINENDI%5D\">";
	content += EMAIL_ADDRESS_CONTACT;
	content += "</a><br/><br/>";

	if (PVParallelView::egl_support()) {
		content += "<br/><b>OpenGL® support:</b><br/>" + PVParallelView::opengl_version();
		content += "<br/><b>EGL™ support:</b><br/>" + PVParallelView::egl_vendor();
	} else {
		content += "<br/>No EGL™/OpenGL® support; using software fallback";
	}
	if (auto openclver = PVOpenCL::opencl_version(); not openclver.empty()) {
		content += "<br/><b>OpenCL™ support:</b><br/>";
		content += QString::fromStdString(openclver);
	} else {
		content += "<br/>No OpenCL™ support; using software fallback";
	}
	content += "<br/><br/>Qt® version " + QString(QT_VERSION_STR);

	_view3D_layout = new QHBoxLayout();

	if (PVParallelView::egl_support() && false) { // Disabled for now as it crash with Qt 5.15.2
		auto widget3d_maker = [this] {
			auto widget3d = new Qt3DExtras::Qt3DWindow();
			{
				Qt3DCore::QEntity* scene = createScene();

				// Camera
				Qt3DRender::QCamera* camera = widget3d->camera();
				camera->lens()->setPerspectiveProjection(45.0f, 16.0f / 9.0f, 0.1f, 1000.0f);
				camera->setPosition(QVector3D(6.f, 0, 0));
				camera->setViewCenter(QVector3D(0, 0, 0));

				// For camera controls
				// Qt3DExtras::QOrbitCameraController* camController =
				//     new Qt3DExtras::QOrbitCameraController(scene);
				// camController->setLinearSpeed(50.0f);
				// camController->setLookSpeed(180.0f);
				// camController->setCamera(camera);

				widget3d->resize(400, 400);

				widget3d->setRootEntity(scene);
			}
			widget3d->defaultFrameGraph()->setClearColor(
			    palette().color(QPalette::Normal, QPalette::Light));
			return widget3d;
		};

		auto widget3d = widget3d_maker();
		auto windowcontainer = QWidget::createWindowContainer(widget3d);
		windowcontainer->setSizePolicy(QSizePolicy::MinimumExpanding,
		                               QSizePolicy::MinimumExpanding);
		windowcontainer->setMinimumSize(QSize(200, 200));

		struct EvFilter: public QObject
		{
			std::function<bool(QEvent*)> verif;
			std::function<void()> f;
			EvFilter(decltype(verif) ver, decltype(f) fun) : verif(ver), f(fun) {}
			bool eventFilter(QObject* obj, QEvent* event) override
			{
				if (verif(event)) {
					f();
				}
				return QObject::eventFilter(obj, event);
			}
		};
		widget3d->installEventFilter(
		    new EvFilter([](QEvent* event) { return event->type() == QEvent::MouseButtonDblClick; },
		                 [widget3d_maker] {
			                 auto window = widget3d_maker();
			                 window->setModality(Qt::WindowModality::ApplicationModal);
			                 window->installEventFilter(new EvFilter(
			                     [window](QEvent* event) {
				                     if (event->type() == QEvent::Close) {
					                     window->deleteLater(); // to prevent window handle leak and
					                                            // crash at app quit
				                     }
				                     return event->type() == QEvent::KeyPress and
				                            ((QKeyEvent*)event)->key() == Qt::Key_Escape;
			                     },
			                     [window] { window->close(); }));
			                 window->showFullScreen();
		                 }));

		_view3D_layout->addWidget(windowcontainer, 0);
	} else {
		auto logo_icon_label = new QLabel;
		logo_icon_label->setPixmap(QPixmap(":/logo_icon.png"));
		logo_icon_label->setSizePolicy(QSizePolicy::Minimum,
		                               QSizePolicy::Minimum);
		logo_icon_label->setMinimumSize(100, 100);
		_view3D_layout->addWidget(logo_icon_label, 0, Qt::AlignRight);
	}

	auto text = new QLabel(content);
	text->setAlignment(Qt::AlignCenter);
	text->setTextFormat(Qt::RichText);
	text->setTextInteractionFlags(Qt::TextBrowserInteraction);
	text->setOpenExternalLinks(true);

	QPushButton* ok = new QPushButton("OK");

	QPushButton* crash = new QPushButton("&Crash ☠");
	crash->setToolTip("Generates a crash of the application in order to test the crash reporter");
	connect(crash, &QPushButton::clicked, [](){
		// taken from Google Breakpad
		volatile int* a = reinterpret_cast<volatile int*>(NULL);
  		*a = 1;
	});

	QGridLayout* software_layout = new QGridLayout;
	software_layout->setHorizontalSpacing(0);
	software_layout->addLayout(_view3D_layout, 0, 0);
	software_layout->addWidget(text, 1, 0);
	software_layout->addWidget(crash, 2, 1);

	QWidget* tab_software = new QWidget;
	tab_software->setLayout(software_layout);
	_tab_widget = new QTabWidget();
	_tab_widget->addTab(tab_software, "Software");
	_changelog_tab = new PVChangeLogWidget;
	_tab_widget->addTab(_changelog_tab, "Changelog");
	_tab_widget->addTab(new PVOpenSourceSoftwareWidget, "Open source software");

	main_layout->addWidget(_tab_widget, 0, 0);
	main_layout->addWidget(ok, 1, 0);

	setLayout(main_layout);

	connect(ok, &QAbstractButton::clicked, this, &QDialog::accept);

	resize(520, 550);

	select_tab(tab);
}

void PVGuiQt::PVAboutBoxDialog::select_tab(Tab tab)
{
	_tab_widget->setCurrentIndex(tab);
}


#ifndef BIG_TEST_OBJS_H
#define BIG_TEST_OBJS_H

#include <pvhive/PVActor.h>

#include <QDialog>
#include <QMessageBox>
#include <QHBoxLayout>
#include <QTimer>
#include <QLabel>

class Entity
{
public:
	Entity(int id) : _id(id)
	{}

	int get_id() const
	{
		return _id;
	}

	void set_value(int v)
	{
		_v = v;
	}

	int get_value() const
	{
		return _v;
	}

private:
	int _id;
	int _v;
};

class EntityGUIActor : public QDialog, public PVHive::PVActor<Entity>
{
	Q_OBJECT

public:
	EntityGUIActor(int id, QString &ent_name, QWidget *parent) : QDialog(parent),
	                                                             _id(id), _value(0)
	{
		int duration = random() % 2000;

		QVBoxLayout *vb = new QVBoxLayout(this);

		QLabel *l = new QLabel(QString("update each ") + QString::number(duration) + " ms");
		vb->addWidget(l);

		_vl = new QLabel("count: NA");
		vb->addWidget(_vl);

		_timer = new QTimer(this);
		connect(_timer, SIGNAL(timeout()), this, SLOT(do_update()));
		_timer->start(duration);

		setWindowTitle("actor " + QString::number(id) + " - " + ent_name);

		setAttribute( Qt::WA_DeleteOnClose, true );

		resize(256, 64);
	}

	~EntityGUIActor()
	{
		std::cout << "death of "<< windowTitle().toStdString() << std::endl;
	}

	int get_id() const
	{
		return _id;
	}

private slots:
	void do_update()
	{
		_vl->setText("count: " + QString::number(_value));
		PVACTOR_CALL(*this, &Entity::set_value, _value);
		++_value;
	}

private:
	int     _id;
	int     _value;
	QTimer *_timer;
	QLabel *_vl;
};

class EntityGUIObserver : public QDialog, public PVHive::PVObserver<Entity>
{
	Q_OBJECT

public:
	EntityGUIObserver(int id, QString &obj_name, QWidget *parent) : QDialog(parent),
	                                                                _id(id)
	{
		QVBoxLayout *vb = new QVBoxLayout(this);

		_vl = new QLabel("value: NA");
		vb->addWidget(_vl);

		setWindowTitle("observer " + QString::number(id) + " - " + obj_name);

		setAttribute( Qt::WA_DeleteOnClose, true );

		resize(256, 64);
	}

	int get_id() const
	{
		return _id;
	}

	~EntityGUIObserver()
	{
		std::cout << "death of "<< windowTitle().toStdString() << std::endl;
	}

	void refresh()
	{
		_vl->setText("value: " + QString::number(get_object()->get_value()));
	}

	void about_to_be_deleted()
	{
		QMessageBox box;
		box.setText("Sure to quit observer " + QString::number(_id) + "?");
		box.setStandardButtons(QMessageBox::Ok);
		box.exec();
		std::cout << "suicide of "<< windowTitle().toStdString() << std::endl;
		close();
	}

private:
	int     _id;
	QLabel *_vl;
};

#endif // BIG_TEST_OBJS_H

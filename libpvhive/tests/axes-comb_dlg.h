
#ifndef AXES_COMB_H
#define AXES_COMB_H

#include <QObject>
#include <QDialog>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>

class TestDlg : public QDialog
{
	Q_OBJECT
public:
	TestDlg(QWidget *parent) :
		QDialog(parent)
	{
		QHBoxLayout* layout = new QHBoxLayout();

		_label = new QLabel("POUET", this);
		layout->addWidget(_label);

		QVBoxLayout *vlayout = new QVBoxLayout();
		layout->addLayout(vlayout);

		QPushButton *up = new QPushButton("up");
		vlayout->addWidget(up);
		connect(up, SIGNAL(clicked(bool)), this, SLOT(move_up(bool)));

		QPushButton *down = new QPushButton("down");
		vlayout->addWidget(down);
		connect(down, SIGNAL(clicked(bool)), this, SLOT(move_down(bool)));

		setLayout(layout);

		resize(320,640);
	}

	void init()
	{
	}

private slots:
	void move_up(bool)
	{
	}

	void move_down(bool)
	{
	}

private:
	QLabel *_label;
};

#endif // AXES_COMB_H

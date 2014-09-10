/**
 * \file PVLocalizedSpinBox.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef __PVLOCALIZEDSPINBOX_H__
#define __PVLOCALIZEDSPINBOX_H__

#include <QSpinBox>

namespace PVGuiQt
{

class PVLocalizedSpinBox : public QSpinBox
{
public:
	PVLocalizedSpinBox(QWidget * parent = nullptr) : QSpinBox(parent) {}

protected:
	QString textFromValue(int value) const override
	{
	   return this->locale().toString(value);
	}
};

}


#endif // __PVLOCALIZEDSPINBOX_H__

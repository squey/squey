/*
 * $Id: PVNrawChild.cpp 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 */

#include <pvkernel/rush/PVNrawChild.h>


//PVRush::PVNrawChild::PVNrawChild(PVNraw *parent_)
//{
//	parent = parent_;
//}
//
//PVRush::PVNrawChild::~PVNrawChild()
//{
//
//}

void PVRush::PVNrawChild::append(QStringList list)
{
	table << list;
}

QString PVRush::PVNrawChild::get_value(PVRow row, PVCol col)
{
	return table.at(row)[col];
}

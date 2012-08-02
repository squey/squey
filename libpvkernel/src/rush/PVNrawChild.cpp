/**
 * \file PVNrawChild.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
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

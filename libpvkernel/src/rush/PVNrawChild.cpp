/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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

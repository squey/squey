/**
 * \file matrix.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVMatrix.h>
#include <QString>

#include <pvkernel/core/picviz_assert.h>

#include <iostream>

#include "is_const.h"

int main()
{
	std::cout << "testing ::get_data()" << std::endl;
	{
		PVCore::PVMatrix<int> a(10,10);
		PVCore::PVMatrix<int>::pointer p = a.get_data();
		PV_ASSERT_VALID(p != nullptr);
		PV_VALID(is_const(p), false);
		PVCore::PVMatrix<int>::const_pointer cp = a.get_data();
		PV_ASSERT_VALID(cp != nullptr);
		PV_VALID(is_const(cp), true);
		PV_ASSERT_VALID(cp == p);
	}
	std::cout << "passed" << std::endl;

	std::cout << "testing ::get_data() const" << std::endl;
	{
		PVCore::PVMatrix<int> a(10,10);
		PVCore::PVMatrix<int>::const_pointer cp = a.get_data();
	}
	std::cout << "passed" << std::endl;

	std::cout << "testing size getters" << std::endl;
	{
		PVCore::PVMatrix<int> a(10,20);
		PV_VALID(a.get_height(), 10U);
		PV_VALID(a.get_width(), 20U);
		PV_VALID(a.get_nrows(), 10U);
		PV_VALID(a.get_ncols(), 20U);
	}
	std::cout << "passed" << std::endl;

	std::cout << "testing ::resize_nrows()" << std::endl;
	{
		PVCore::PVMatrix<int> a(10,10);
		PV_VALID(a.get_nrows(), 10U);
		a.resize_nrows(20);
		PV_VALID(a.get_nrows(), 20U);
	}
	std::cout << "passed" << std::endl;

	std::cout << "testing ::resize_ncols()" << std::endl;
	{
		PVCore::PVMatrix<int> a(10,10);
		PV_VALID(a.get_ncols(), 10U);
		a.resize_ncols(20);
		PV_VALID(a.get_ncols(), 20U);
	}
	std::cout << "passed" << std::endl;

	std::cout << "testing ::resize()" << std::endl;
	{
		PVCore::PVMatrix<int> a(10,10);
		PV_VALID(a.get_nrows(), 10U);
		PV_VALID(a.get_ncols(), 10U);
		a.resize(20, 30);
		PV_VALID(a.get_nrows(), 20U);
		PV_VALID(a.get_ncols(), 30U);
	}
	std::cout << "passed" << std::endl;

	std::cout << "testing ::free()" << std::endl;
	{
		PVCore::PVMatrix<int> a(10,10);
		a.free();
		PV_VALID(a.get_ncols(), 0U);
		PV_VALID(a.get_nrows(), 0U);
		PV_ASSERT_VALID(a.get_data() == nullptr);
	}
	std::cout << "passed" << std::endl;

	// fill matrix
	PVCore::PVMatrix<float> mf;
	mf.resize(10, 10);
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			float v = (float)i + ((float)j/10.0f);
			mf.at(i,j) = v;
		}
	}

	std::cout << "testing ::at()" << std::endl;
	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			float v = (float)i + ((float)j/10.0f);
			PV_VALID(mf.at(i,j), v, "i", i, "j", j);
		}
	}
	std::cout << "passed" << std::endl;

	std::cout << "testing ::transpose_to()" << std::endl;
	{
		PVCore::PVMatrix<float> tmf;
		mf.transpose_to(tmf);

		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 10; j++) {
				PV_VALID(tmf.at(i,j), mf.at(j,i), "i", i, "j", j);
			}
		}
	}
	std::cout << "passed" << std::endl;

	std::cout << "testing ::copy_to()" << std::endl;
	{
		PVCore::PVMatrix<float> cmf;
		mf.copy_to(cmf);
		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 10; j++) {
				PV_VALID(cmf.at(i,j), mf.at(i,j), "i", i, "j", j);
			}
		}
	}
	std::cout << "passed" << std::endl;

	std::cout << "testing ::swap()" << std::endl;
	{
		PVCore::PVMatrix<float> tmf;
		PVCore::PVMatrix<float> dmf;

		mf.copy_to(tmf);
		dmf.swap(tmf);

		for (int i = 0; i < 10; i++) {
			for (int j = 0; j < 10; j++) {
				PV_VALID(dmf.at(i,j), mf.at(i,j), "i", i, "j", j);
			}
		}
	}
	std::cout << "passed" << std::endl;


	std::cout << "testing ::column's size getter" << std::endl;
	{
		PVCore::PVMatrix<int> a(10,10);
		for (int i = 0; i < 10; i++) {
			PV_VALID(a.get_col(i).size(), 10U, "i", i);
		}
	}
	std::cout << "passed" << std::endl;

	std::cout << "testing ::column::at()" << std::endl;
	for (size_t j = 0; j < mf.get_ncols(); j++) {
		PVCore::PVMatrix<float>::column col = mf.get_col(j);
		for (size_t i = 0; i < col.size(); i++) {
			PV_VALID(col.at(i), mf.at(i,j), "i", i, "j", j);
		}
	}
	std::cout << "passed" << std::endl;

	std::cout << "testing ::line's size getter" << std::endl;
	{
		PVCore::PVMatrix<int> a(10,10);
		for (int i = 0; i < 10; i++) {
			PV_VALID(a.get_row(i).size(), 10U, "i", i);
		}
	}
	std::cout << "passed" << std::endl;

	std::cout << "testing ::line::at()" << std::endl;
	for (size_t j = 0; j < mf.get_nrows(); j++) {
		PVCore::PVMatrix<float>::line row = mf.get_row(j);
		for (size_t i = 0; i < row.size(); i++) {
			PV_VALID(row.at(i), mf.at(j,i), "i", i, "j", j);
		}
	}
	std::cout << "passed" << std::endl;

	std::cout << "testing data preservation while resizing (using strings as elements)" << std::endl;
	{
		QString str("salut");
		PVCore::PVMatrix<QString> b;
		b.resize(10,10,str);
		b.resize_nrows(40, QString());
		b.resize_nrows(60, QString());

		for (size_t i = 0; i < b.get_nrows(); ++i) {
			for (size_t j = 0; j < b.get_ncols(); ++j) {
				if ((i < 10) && (j < 10)) {
					PV_VALID(b.at(i,j).toStdString(), str.toStdString(), "i", i, "j", j);
				} else {
					PV_VALID(b.at(i,j).toStdString(), QString().toStdString(), "i", i, "j", j);
				}
			}
		}
	}
	std::cout << "passed" << std::endl;

	return 0;
}

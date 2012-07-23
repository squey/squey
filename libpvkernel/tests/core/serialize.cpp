/**
 * \file serialize.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/core/PVSerializeArchiveOptions.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <iostream>
#include <QCoreApplication>

#include <boost/enable_shared_from_this.hpp>

#include <list>

class PVTestChild: public boost::enable_shared_from_this<PVTestChild>
{
	friend class PVCore::PVSerializeObject;
public:
	PVTestChild():
		_a(0)
	{ }
	PVTestChild(int a):
		_a(a)
	{ }

public:
	void dump() const
	{
		std::cout << "child int: " << _a << std::endl;
	}

protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*version*/)
	{
		so.attribute("int", _a, (int) 0);
	}
private:
	int _a;
};

class PVTestBuf
{
	friend class PVCore::PVSerializeObject;
public:
	PVTestBuf()
	{
		buf[0] = buf[1] = buf[2] = buf[3] = 0;
	}
	void set_buf()
	{
		buf[0] = buf[1] = buf[2] = buf[3] = 10;
	}
	void dump() const
	{
		std::cout << "buf:" << std::endl;
		for (int i = 0; i < 4; i++) {
			std::cout << buf[i] << std::endl;
		}
	}
protected:
	void serialize_write(PVCore::PVSerializeObject& so)
	{
		so.buffer("data", &buf, sizeof(int)*4);
	}

	void serialize_read(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*version*/)
	{
		so.buffer("data", &buf, sizeof(int)*4);
	}

	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*version*/)
	{
		so.split(*this);
	}
protected:
	int buf[4];
};

class PVTestObj
{
	friend class PVCore::PVSerializeObject;
public:
	PVTestObj():
		_a(0)
	{ }

	PVTestObj(QString const& str, int a):
		_str(str),
		_a(a),
		_child(10*a)
	{
		for (int i = 0; i < 10; i++) {
			_list_children.push_back(PVTestChild(i));
		}
		for (int i = 0; i < 10; i++) {
			_list_ints.push_back(i);
		}
		_buf.set_buf();

		boost::shared_ptr<PVTestChild> p1(new PVTestChild(1));
		boost::shared_ptr<PVTestChild> p2(new PVTestChild(2));
		_list_p1.push_back(p1);
		_list_p1.push_back(p2);
		_list_p2.push_back(p1);
		_list_p2.push_back(p2);
	}

public:
	void dump() const
	{
		std::cout << "str: " << qPrintable(_str) << std::endl;
		std::cout << "int: " << _a << std::endl;
		_child.dump();
		std::cout << "int list has " << _list_ints.size() << " elements" << std::endl;
		std::list<int>::const_iterator it_i;
		for (it_i = _list_ints.begin(); it_i != _list_ints.end(); it_i++) {
			std::cout << *it_i << std::endl;
		}
		std::cout << "child list has " << _list_children.size() << " elements" << std::endl;
		std::list<PVTestChild>::const_iterator it;
		for (it = _list_children.begin(); it != _list_children.end(); it++) {
			it->dump();
		}
		_buf.dump();
		std::cout << "references:" << std::endl;
		std::cout << "-----------" << std::endl;
		std::cout << "Original list:" << std::endl;
		std::list<boost::shared_ptr<PVTestChild> >::const_iterator it_ptr;
		for (it_ptr = _list_p1.begin(); it_ptr != _list_p1.end(); it_ptr++) {
			std::cout << it_ptr->get() << std::endl;
		}
		std::cout << "Ref list:" << std::endl;
		for (it_ptr = _list_p2.begin(); it_ptr != _list_p2.end(); it_ptr++) {
			std::cout << it_ptr->get() << std::endl;
		}
	}
protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t /*version*/)
	{
		so.attribute("str", _str, QString("default"));
		so.attribute("int", _a, (int) 0);
		so.object("child", _child, "That's a child", true);
		so.list("children", _list_children);
		so.list_attributes("int_children", _list_ints);
		so.object("buf_test", _buf);

		PVTestChild a(1);
		PVCore::PVSerializeObject_p org = so.list("ref_org", _list_p1, QString(), &a);
		so.list_ref("test_ref", _list_p2, org);
	}

private:
	QString _str;
	int _a;
	PVTestChild _child;
	std::list<PVTestChild> _list_children;
	std::list<boost::shared_ptr<PVTestChild> > _list_p1;
	std::list<boost::shared_ptr<PVTestChild> > _list_p2;
	std::list<int> _list_ints;
	PVTestBuf _buf;
};

int main(int argc, char** argv)
{
	QCoreApplication(argc, argv);

	// Main object
	PVTestObj obj("salut", 4);

	// Options
	PVCore::PVSerializeArchiveOptions_p options(new PVCore::PVSerializeArchiveOptions(1));
	options->get_root()->object("obj", obj);
	options->get_root()->get_child_by_name("obj")->get_child_by_name("child")->set_write(false);

	PVCore::PVSerializeArchive_p ar(new PVCore::PVSerializeArchive("/tmp/test", PVCore::PVSerializeArchive::write, 1));
	ar->set_options(options);
	ar->get_root()->object("obj", obj);


	// Get it back
	ar = PVCore::PVSerializeArchive_p(new PVCore::PVSerializeArchive("/tmp/test", PVCore::PVSerializeArchive::read, 1));
	PVTestObj obj_r;
	obj_r.dump();
	ar->get_root()->object("obj", obj_r);
	obj_r.dump();

	// Same with zip archives
	ar = PVCore::PVSerializeArchive_p(new PVCore::PVSerializeArchiveZip("/tmp/testarch.pv", PVCore::PVSerializeArchive::write, 1));
	ar->get_root()->object("obj", obj);
	ar->finish();

	// Get it back
	ar = PVCore::PVSerializeArchive_p(new PVCore::PVSerializeArchiveZip("/tmp/testarch.pv", PVCore::PVSerializeArchive::read, 1));
	PVTestObj obj_r2;
	ar->get_root()->object("obj", obj_r2);
	ar->finish();
	obj_r2.dump();

	return 0;
}

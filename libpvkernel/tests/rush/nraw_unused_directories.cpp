
#include <pvkernel/core/picviz_assert.h>
#include <pvkernel/core/PVDirectory.h>

#include <pvkernel/rush/PVNraw.h>

#include <unistd.h>

#include <QDirIterator>

#define DIR_PATTERN "test_nraw_unused_dir-XXXXXX"
#define DIR_REGEXP  "test_nraw_unused_dir-??????"

void file_touch(const QString &fname)
{
	QFile f(fname);
	f.open(QIODevice::WriteOnly);
	f.close();
}


void dump_lists(const QStringList &all_dirs, const QStringList &unused_dirs)
{
	std::cout << "all dirs :";
	for (const QString &value : all_dirs) {
		std::cout << " " << value.toLocal8Bit().constData() << std::endl;
	}
	std::cout << std::endl;

	std::cout << "unused dirs :";
	for (const QString &value : unused_dirs) {
		std::cout << " " << value.toLocal8Bit().constData() << std::endl;
	}
	std::cout << std::endl;
}

int main(int argc, char** argv)
{
	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " path_nraw" << std::endl;
		return 1;
	}

	QString root_dir_name = QString(argv[1]) + QDir::separator() + "test_nraw_unused_dir";
	QDir root_dir(root_dir_name);

	PVCore::PVDirectory::remove_rec(root_dir.absolutePath());

	root_dir.mkpath(root_dir_name);

	QStringList all_dirs;
	QString file_to_open;
	QString dir_to_block;

	std::cout << "creating random directories and files" << std::endl;
	for(int i = 0; i < 10; ++i) {
		QString d = PVCore::PVDirectory::temp_dir(root_dir,
		                                          DIR_PATTERN);

		all_dirs << d;
		root_dir.mkpath(d);

		for(int j = 0; j < 10; ++j) {
			QString f(d + "/aa-" + QString::number(j));

			if ((i == 3) && (j == 6)) {
				file_to_open = f;
				dir_to_block = d;
			}

			file_touch(f);
		}
	}

	all_dirs.sort();

	std::cout << "testing when no files are opened" << std::endl;

	QStringList unused_dirs =
		PVRush::PVNraw::list_unused_nraw_directories(root_dir.absolutePath(),
		                                             DIR_REGEXP);
	unused_dirs.sort();

	PV_ASSERT_VALID(all_dirs == unused_dirs);

	std::cout << "testing with one opened file ("
	          << qPrintable(file_to_open)
	          << ")" << std::endl;

	QFile f(file_to_open);
	f.open(QIODevice::WriteOnly);

	char c;

	f.write(&c, 1);

	unused_dirs =
		PVRush::PVNraw::list_unused_nraw_directories(root_dir.absolutePath(),
		                                             DIR_REGEXP);
	unused_dirs.sort();

	PV_ASSERT_VALID(all_dirs != unused_dirs);

	std::cout << "checking that unused dirs + used dirs == all dirs" << std::endl;

	QFileInfo fi(file_to_open);
	unused_dirs << dir_to_block;

	unused_dirs.sort();

	PV_ASSERT_VALID(all_dirs == unused_dirs);

	f.close();

	std::cout << "removing unused directories" << std::endl;

	PVRush::PVNraw::remove_unused_nraw_directories(root_dir.absolutePath(),
	                                               DIR_REGEXP);


	std::cout << "checking there is no more unused directories" << std::endl;

	unused_dirs =
		PVRush::PVNraw::list_unused_nraw_directories(root_dir.absolutePath(),
		                                             DIR_REGEXP);

	PV_ASSERT_VALID(unused_dirs.isEmpty());

	std::cout << "cleaning test root dir" << std::endl;

	PVCore::PVDirectory::remove_rec(root_dir_name);


	return 0;
}

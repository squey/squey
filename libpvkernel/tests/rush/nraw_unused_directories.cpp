
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

	return 0;
}

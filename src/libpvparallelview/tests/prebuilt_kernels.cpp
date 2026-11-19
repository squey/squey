/* * MIT License
 *
 * © Squey, 2026
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// Kernels built in advance are loaded from where POCL_KERNEL_CACHE_PREBUILT_DIR
// says, and POCL_KERNEL_CACHE_PREBUILT_ONLY never has PortableCL build one: the
// App Store version of macOS may load no code it writes itself.
//
// PortableCL reads its environment once per process, so each case runs in a
// process of its own: this test, run again with the role of the case.

#include <pvparallelview/PVBCIDrawingBackendOpenCL.h>
#include <pvparallelview/PVParallelView.h>

#include <pvkernel/core/squey_assert.h>

#include <QCoreApplication>
#include <QDir>
#include <QDirIterator>
#include <QProcess>
#include <QProcessEnvironment>
#include <QTemporaryDir>

#include <map>
#include <string>

namespace
{

constexpr int DRAWN_BY_OPENCL = 0;
constexpr int DRAWN_BY_QPAINTER = 3;

int draw()
{
	PVParallelView::common::RAII_backend_init resources;
	return &PVParallelView::common::backend() == &PVParallelView::PVBCIDrawingBackendOpenCL::get()
	           ? DRAWN_BY_OPENCL
	           : DRAWN_BY_QPAINTER;
}

/**
 * The work-group functions PortableCL linked under dir, which it names after
 * their kernel with the extension of a shared library.
 */
size_t kernel_binaries_in(const QString& dir)
{
#ifdef _WIN32
	const QString binary_filter = "*.dll";
#else
	const QString binary_filter = "*.so";
#endif
	size_t count = 0;
	QDirIterator it(dir, {binary_filter}, QDir::Files, QDirIterator::Subdirectories);
	while (it.hasNext()) {
		it.next();
		++count;
	}
	return count;
}

/**
 * Runs this test again in the given role, on the CPU device: a GPU driver builds
 * its kernels on its own, out of reach of the variables under test.
 */
int run_as(const QStringList& role, const std::map<QString, QString>& variables)
{
	QProcessEnvironment environment = QProcessEnvironment::systemEnvironment();
	environment.insert("FORCE_CPU", "1");
	for (const auto& [name, value] : variables) {
		environment.insert(name, value);
	}

	QProcess process;
	process.setProcessEnvironment(environment);
	process.setProcessChannelMode(QProcess::ForwardedChannels);
	process.start(QCoreApplication::applicationFilePath(), role);
	if (not process.waitForFinished(-1) or process.exitStatus() != QProcess::NormalExit) {
		return -1;
	}
	return process.exitCode();
}

} // namespace

int main(int argc, char** argv)
{
	if (argc == 3 and std::string(argv[1]) == "prepare") {
		return PVParallelView::PVBCIDrawingBackendOpenCL::prepare_prebuilt_kernels(argv[2]) ? 0 : 1;
	}
	if (argc == 2 and std::string(argv[1]) == "draw") {
		return draw();
	}

	QCoreApplication app(argc, argv);

	QTemporaryDir root;
	PV_ASSERT_VALID(root.isValid(), "temporary directory", root.errorString().toStdString());

	const QString prebuilt = root.filePath("prebuilt");
	PV_VALID(run_as({"prepare", prebuilt}, {}), 0);
	const size_t prebuilt_count = kernel_binaries_in(prebuilt);
	PV_ASSERT_VALID(prebuilt_count > 0, "kernel binaries prepared", prebuilt_count);

	// All there, loaded from there: nothing gets built into the cache.
	const QString cache_all_prebuilt = root.filePath("cache_all_prebuilt");
	PV_VALID(run_as({"draw"}, {{"POCL_CACHE_DIR", cache_all_prebuilt},
	                           {"POCL_KERNEL_CACHE_PREBUILT_DIR", prebuilt},
	                           {"POCL_KERNEL_CACHE_PREBUILT_ONLY", "1"}}),
	         DRAWN_BY_OPENCL);
	PV_VALID(kernel_binaries_in(cache_all_prebuilt), size_t(0));

	// None there, and none to be built: the QPainter backend draws instead.
	const QString none = root.filePath("none");
	PV_ASSERT_VALID(QDir().mkpath(none), "directory", none.toStdString());
	const QString cache_none_prebuilt = root.filePath("cache_none_prebuilt");
	PV_VALID(run_as({"draw"}, {{"POCL_CACHE_DIR", cache_none_prebuilt},
	                           {"POCL_KERNEL_CACHE_PREBUILT_DIR", none},
	                           {"POCL_KERNEL_CACHE_PREBUILT_ONLY", "1"}}),
	         DRAWN_BY_QPAINTER);
	PV_VALID(kernel_binaries_in(cache_none_prebuilt), size_t(0));

	// None there, but allowed to be built: they are, and into the cache, where
	// the two cases above would have found them.
	const QString cache_built = root.filePath("cache_built");
	PV_VALID(run_as({"draw"},
	                {{"POCL_CACHE_DIR", cache_built}, {"POCL_KERNEL_CACHE_PREBUILT_DIR", none}}),
	         DRAWN_BY_OPENCL);
	PV_VALID(kernel_binaries_in(cache_built), prebuilt_count);

	return 0;
}

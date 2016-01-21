#ifndef LIBINENDI_TESTS_COMMON_H
#define LIBINENDI_TESTS_COMMON_H

#include "test-env.h"

#include <inendi/common.h>
#include <inendi/PVView.h>
#include <inendi/PVRoot.h>
#include <inendi/PVScene.h>
#include <inendi/PVSource.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVPluginsLoad.h>

#include <QCoreApplication>

namespace pvtest {

    /**
     * Get a tmp filename not already use.
     *
     * @warning, It can be use between this call and your creation.
     */
    std::string get_tmp_filename()
    {
        std::string out_path;
        // Duplicate input log to make it bigger
        out_path.resize(L_tmpnam);
        // We assume that this name will not be use by another program before we create it.
        tmpnam (&out_path.front());

        return out_path;
    }

    /**
     * Create and save context for a view creation.
     *
     * * Required when we want to work with NRaw content
     */
    class TestEnv
    {

        public:
        /**
         * Initialize Inspector internal until a view is correctly build and return this view.
         */
        TestEnv(std::string const& log_file, std::string const& format_file)
        {
            // Need this core application to find plugins path.
            std::string prog_name = "test_inendi";
            char* arg = const_cast<char*>(prog_name.c_str());
            int argc = 1;
            QCoreApplication app(argc, &arg);

            init_env();

            // Load plugins to fill the nraw
            PVFilter::PVPluginsLoad::load_all_plugins(); // Splitters
            PVRush::PVPluginsLoad::load_all_plugins(); // Sources

            // Initialize sse4 detection
            PVCore::PVIntrinsics::init_cpuid();

            //Input file
            QString path_file = QString::fromStdString(log_file);
            PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_file));

            // Load the given format file
            QString path_format = QString::fromStdString(format_file);
            PVRush::PVFormat format("format", path_format);
            if (!format.populate()) {
                throw std::runtime_error("Can't read format file " + format_file);
            }

            // Get the source creator
            PVRush::PVSourceCreator_p sc_file;
            if (!PVRush::PVTests::get_file_sc(file, format, sc_file)) {
                throw std::runtime_error("Can't get sources.");
            }

            // Create the PVSource object
            Inendi::PVScene_p scene(root, "scene");
            Inendi::PVSource_p src(scene, PVRush::PVInputType::list_inputs() << file, sc_file, format);
            PVRush::PVControllerJob_p job = src->extract();
            job->wait_end();

            // Map the nraw
            Inendi::PVMapped_p mapped(src);
            mapped->process_from_parent_source();

            // And plot the mapped values
            Inendi::PVPlotted_p plotted(mapped);
            plotted->process_from_parent_mapped();

            view = src->current_view();
        }

        Inendi::PVRoot_p root;
        Inendi::PVView* view;
    };

}

#endif

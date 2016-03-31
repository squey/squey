#ifndef LIBPVKERNEL_RUSH_TESTS_COMMON_H
#define LIBPVKERNEL_RUSH_TESTS_COMMON_H

#include "test-env.h"

#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVInputDescription.h>
#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <pvkernel/rush/PVUnicodeSource.h>

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
     * Prepare Context to run tests.
     *
     * * Set environment variables
     * * Prepare QCoreApplication
     * * Load plugins
     * * Init cpu features
     * * Duplicate log file
     */
    std::string init_ctxt(std::string const& log_file, size_t dup)
    {
            // Need this core application to find plugins path.
            std::string prog_name = "test_pvkernel_rush";
            char* arg = const_cast<char*>(prog_name.c_str());
            int argc = 1;
            QCoreApplication app(argc, &arg);

            init_env();

            // Load plugins to fill the nraw
            PVFilter::PVPluginsLoad::load_all_plugins(); // Splitters
            PVRush::PVPluginsLoad::load_all_plugins(); // Sources

            // Initialize sse4 detection
            PVCore::PVIntrinsics::init_cpuid();

            std::ifstream ifs(log_file);
            std::string content{std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>()};

            std::string big_log_file = get_tmp_filename();
            std::ofstream big_file(big_log_file);
            // Duplicate file to have one millions lines
            for(size_t i=0; i<dup; i++) {
                big_file << content;
            }
            return big_log_file;
    }

    /**
     * Create sources and splitter to be read for splitting.
     *
     * Usefull to check splitter behavior.
     */
    class TestSplitter
    {
        public:
        TestSplitter(std::string const& log_file, size_t dup = 1):
                _big_file_path(init_ctxt(log_file, dup)),
                _source(std::make_shared<PVRush::PVInputFile>(_big_file_path.c_str()), chunk_size)
        {}

        PVRush::PVUnicodeSource<>& get_source() { return _source; }

        ~TestSplitter() { std::remove(_big_file_path.c_str()); }

        private:
            static constexpr size_t chunk_size = 6000;

            std::string _big_file_path;
            PVRush::PVUnicodeSource<> _source;
    };

    /**
     * Create and save context for a view creation.
     *
     * * Required when we want to work with NRaw content
     */
    class TestEnv
    {

        public:
        /**
         * Initialize Inspector internal until pipeline is ready to process inputs.
         *
         * NRaw is not loaded, it has to be done with the load_data methods.
         */
        TestEnv(std::string const& log_file, std::string const& format_file, size_t dup = 1):
                _format("format", QString::fromStdString(format_file)),
                _big_file_path(init_ctxt(log_file, dup))
        {
            //Input file
            QString path_file = QString::fromStdString(_big_file_path);
            PVRush::PVInputDescription_p file(new PVRush::PVFileDescription(path_file));

            // Load the given format file
            if (!_format.populate()) {
                throw std::runtime_error("Can't read format file " + format_file);
            }

            // Get the source creator
            PVRush::PVSourceCreator_p sc_file;
            if (!PVRush::PVTests::get_file_sc(file, _format, sc_file)) {
                throw std::runtime_error("Can't get sources.");
            }

            // Process that file with the found source creator thanks to the extractor
            PVRush::PVSourceCreator::source_p src = sc_file->create_source_from_input(file, _format);
            if (!src) {
                throw std::runtime_error("Unable to create PVRush source from file " + log_file + "\n");
            }

            // Create the extractor
            _ext.add_source(src);
            _ext.set_format(_format);
            _ext.set_chunk_filter(_format.create_tbb_filters());
        }

        void load_data(size_t nb_lines) {
            PVRush::PVControllerJob_p job = _ext.process_from_agg_nlines(0, nb_lines);
            job->wait_end();
        }

        /**
         * Clean up duplicated file when it is over.
         */
        ~TestEnv()
        {
            std::remove(_big_file_path.c_str());
        }

        /**
         * Get number of row in the imported NRaw.
         */
        size_t get_nraw_size() const { return _ext.get_nraw().get_row_count(); }

        PVRush::PVExtractor _ext;

    private:
        PVRush::PVFormat _format;
        std::string _big_file_path;


    };

}

#endif

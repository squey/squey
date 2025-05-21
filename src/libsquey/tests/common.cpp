#include "common.h"

std::string pvtest::get_tmp_filename()
{
	char buffer[L_tmpnam];
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
	std::string tmp_filename = tmpnam(buffer);
#pragma GCC diagnostic pop
	return tmp_filename;
}

pvtest::TestEnv::TestEnv(
    std::vector<std::string> const& log_files,
    std::string const& format_file,
    size_t dup /*= 1*/,
    ProcessUntil until /*= ProcessUntil::Source*/,
    const std::string& nraw_loading_from_disk_dir /*= ""*/)
{
    // Need this core application to find plugins path.
    std::string prog_name = "test_squey";
    char* arg = const_cast<char*>(prog_name.c_str());
    int argc = 1;
    QCoreApplication app(argc, &arg);

    init_env();

    import(log_files, format_file, dup, true, nraw_loading_from_disk_dir);

    switch (until) {
    case ProcessUntil::Source:
        return;
    case ProcessUntil::Mapped:
        compute_mappings();
        return;
    case ProcessUntil::Scaled:
        compute_mappings();
        compute_scalings();
        return;
    case ProcessUntil::View:
        compute_mappings();
        compute_scalings();
        compute_views();
        return;
    }
}

Squey::PVSource& pvtest::TestEnv::import(
    std::vector<std::string> const& log_files,
    std::string const& format_file,
    size_t dup,
    bool new_scene /* = true*/,
    const std::string& nraw_loading_from_disk_dir /*= ""*/)
{

    if (dup != 1 and log_files.size() > 1) {
        throw std::runtime_error("We don't handle multiple input with duplication");
    }

    std::string new_path = log_files[0];
    if (dup > 1) {
        new_path = get_tmp_filename();
        _big_file_paths.push_back(new_path);
        std::ifstream ifs(std::filesystem::path{log_files[0]});
        std::string content{std::istreambuf_iterator<char>(ifs),
                            std::istreambuf_iterator<char>()};

        std::ofstream big_file{std::filesystem::path(new_path)};
        // Duplicate file to have one millions lines
        for (size_t i = 0; i < dup; i++) {
            big_file << content;
        }
    }

    PVRush::PVInputType::list_inputs inputs;

    // Input file
    QString path_file = QString::fromUtf8(new_path);
    PVRush::PVInputDescription_p file(
        new PVRush::PVFileDescription(path_file, log_files.size() > 1));
    inputs << file;

    for (size_t i = 1; i < log_files.size(); i++) {
        inputs << PVRush::PVInputDescription_p(new PVRush::PVFileDescription(
            QString::fromUtf8(log_files[i]), log_files.size() > 1));
    }

    // Load the given format file
    QString path_format = QString::fromUtf8(format_file);
    PVRush::PVFormat format("format", path_format);

    // Get the source creator
    PVRush::PVSourceCreator_p sc_file;
    if (!PVRush::PVTests::get_file_sc(file, format, sc_file)) {
        throw std::runtime_error("Can't get sources.");
    }

    // Create the PVSource object
    Squey::PVScene* scene =
        (new_scene) ? &root.emplace_add_child("scene") : root.get_children().front();
    Squey::PVSource& src = scene->emplace_add_child(inputs, sc_file, format);

    if (not nraw_loading_from_disk_dir.empty()) {
        src.get_rushnraw().load_from_disk(nraw_loading_from_disk_dir);
    } else {
        PVRush::PVControllerJob_p job = src.extract(0);
        src.wait_extract_end(job);
    }
    return src;
}
kind: manual

sources:
  - kind: local
    path: buildstream/files

build-depends:
- base.bst
# - squey-cleanup.bst

config:
  install-commands:
  - install -D -m 0644 "org.squey.Squey.png" -t "%{install-root}%{prefix}/share/icons/hicolor/128x128/apps/"
  - install -D -m 0644 "org.squey.Squey.png" -t "%{install-root}%{prefix}/share/app-info/icons/flatpak/128x128/"
  - |
    cat <<EOF >org.squey.Squey.desktop
    [Desktop Entry]
    Type=Application
    Name=Squey
    Exec=squey_launcher.sh %F
    Icon=org.squey.Squey
    Categories=Science;Graphics;Network;
    MimeType=application/org.squey.investigation;text/csv;text/tab-separated-values;application/vnd.apache.parquet;application/vnd.tcpdump.pcap;application/x-pcapng;
    EOF
    install -D -m 0644 "org.squey.Squey.desktop" -t "%{install-root}%{prefix}/share/applications"
  - |
    # What the desktop entry above names has to exist in the shared MIME
    # database, or the association is written against a type no file is ever
    # given. Declared here are the types that database does not carry -- the
    # investigation is ours, and Parquet and pcapng reached it late enough that
    # a supported distribution may still be without them -- and the compressed
    # spellings it otherwise resolves to their container: *.csv.gz is longer
    # than *.gz, so the glob below wins and the file is offered to Squey.
    #
    # A type that is already known keeps its own comment: re-stating it here
    # would replace a translated description with an English one.
    cat <<EOF >org.squey.Squey.xml
    <?xml version="1.0" encoding="UTF-8"?>
    <mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
        <mime-type type="application/org.squey.investigation">
            <comment>Squey investigation</comment>
            <glob pattern="*.pvi"/>
        </mime-type>
        <mime-type type="application/vnd.apache.parquet">
            <comment>Apache Parquet file</comment>
            <glob pattern="*.parquet"/>
        </mime-type>
        <mime-type type="application/x-pcapng">
            <comment>PCAPNG capture file</comment>
            <glob pattern="*.pcapng"/>
        </mime-type>
        <mime-type type="text/csv">
            <glob pattern="*.csv.zip"/>
            <glob pattern="*.csv.gz"/>
            <glob pattern="*.csv.bz2"/>
            <glob pattern="*.csv.xz"/>
            <glob pattern="*.csv.zst"/>
        </mime-type>
        <mime-type type="text/tab-separated-values">
            <glob pattern="*.tsv.zip"/>
            <glob pattern="*.tsv.gz"/>
            <glob pattern="*.tsv.bz2"/>
            <glob pattern="*.tsv.xz"/>
            <glob pattern="*.tsv.zst"/>
        </mime-type>
    </mime-info>
    EOF
    install -D -m 0644 "org.squey.Squey.xml" -t "%{install-root}%{prefix}/share/mime/packages"
  - install -D -m 0644 "org.squey.Squey.metainfo.xml" -t "%{install-root}%{prefix}/share/metainfo"

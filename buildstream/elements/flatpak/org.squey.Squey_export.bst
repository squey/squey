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
    MimeType=application/org.squey.investigation;text/csv;text/tab-separated-values;application/vnd.apache.parquet;application/vnd.tcpdump.pcap;application/vnd.tcpdump.pcapng;
    EOF
    install -D -m 0644 "org.squey.Squey.desktop" -t "%{install-root}%{prefix}/share/applications"
  - |
    cat <<EOF >squey-mime.xml
    <mime-info xmlns="http://www.freedesktop.org/standards/shared-mime-info">
        <mime-type type="application/org.squey.investigation">
            <comment>Squey investigation</comment>
            <glob pattern="*.pvi"/>
        </mime-type>
        <mime-type type="text/csv">
            <comment>CSV file</comment>
            <glob pattern="*.csv"/>
            <glob pattern="*.csv.zip"/>
            <glob pattern="*.csv.gz"/>
            <glob pattern="*.csv.bz2"/>
            <glob pattern="*.csv.xz"/>
            <glob pattern="*.csv.zst"/>
        </mime-type>
        <mime-type type="text/tab-separated-values">
            <comment>TSV file</comment>
            <glob pattern="*.tsv"/>
            <glob pattern="*.tsv.zip"/>
            <glob pattern="*.tsv.gz"/>
            <glob pattern="*.tsv.bz2"/>
            <glob pattern="*.tsv.xz"/>
            <glob pattern="*.tsv.zst"/>
        </mime-type>
        <mime-type type="application/vnd.apache.parquet">
            <comment>Apache Parquet file</comment>
            <glob pattern="*.parquet"/>
        </mime-type>
        <mime-type type="application/vnd.tcpdump.pcap">
            <comment>PCAP file</comment>
            <glob pattern="*.pcap"/>
        </mime-type>
        <mime-type type="application/vnd.tcpdump.pcapng">
            <comment>PCAPNG file</comment>
            <glob pattern="*.pcapng"/>
        </mime-type>
    </mime-info>
    install -D -m 0644 "squey-mime.xml" -t "%{install-root}%{prefix}/share/mime/packages"
  - install -D -m 0644 "org.squey.Squey.metainfo.xml" -t "%{install-root}%{prefix}/share/metainfo"

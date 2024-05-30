# About

![](https://gitlab.com/squey/squey/-/raw/main/buildstream/files/screenshots/squey_screenshot.png)

## Project description

<!-- project_description_start -->

<a href="https://squey.org">Squey</a> is designed from the ground up to take advantage of GPUs and CPUs to perform interactive explorations of massive amounts of data.

It gives users an exhaustive yet intuitive multi-view representation of columnar data and can ingest from:
<ol>
    <li>Structured text files (CSV, logs, ...)</li>
    <li>Apache Parquet files</li>
    <li>Pcap files</li>
    <li>SQL databases</li>
    <li>Elasticsearch databases</li>
</ol>

Squey strives to deliver value through its <b>V.I.SU</b> approach:

<ul>
  <li><b>Visualize</b>: Leverage various visual representations of raw data in combination with statistics</li>
  <li><b>Investigate</b>: Use filters to build an accurate understanding of millions of rows while switching instantly between capturing the big picture and focusing on the details</li>
  <li><b>Spot the Unknown</b>: As a structured understanding of the data emerges, identify unknowns and anomalies</li>
</ul>

Squey can be used for many different purposes, such as:
<ul>
    <li><b>BI and Big Data</b>: Bootstrap initial understanding of complex datasets and deep dive where necessary to design accurate data processing</li>
    <li><b>Cybersecurity</b>: Detect weak signals such as attacks and data leaks</li>
    <li><b>IT troubleshooting</b>: Resolve network issues and improve application performance</li>
    <li><b>Machine Learning</b>: Design training dataset to fulfill targeted improvements of Machine Learning models</li>
</ul>

<br>
Give yourself a chance to <b>see</b> your data and have fun exploring!

<!-- project_description_end -->

## Installation

### Linux

<a href='https://flathub.org/apps/details/org.squey.Squey'><img width='190px' alt='Download on Flathub' src='https://flathub.org/assets/badges/flathub-badge-en.png'/></a>

```
flatpak install --user -y https://dl.flathub.org/repo/appstream/org.squey.Squey.flatpakref
```

### Windows

Windows support is available through the use of WSL2:

https://squey.gitlab.io/squey/squey_installer.exe

### Container

Deploying the software as a service in a private cloud:

[![dockeri.co](https://dockerico.blankenship.io/image/squey/squey)](https://hub.docker.com/r/squey/squey)


Or building your own container image:

https://squey.gitlab.io/squey/squey_docker.zip


See ```README.md``` contained in archive file

## Reference manual

https://doc.squey.org

## Development

See the developement [README.md](buildstream/README.md) page.

### Installing a flatpak development branch

Note : Merge Requests having the `action::flatpak_export` [label](https://gitlab.com/squey/squey/-/labels#) are exported by the CI/CD pipeline as a flatpak package named after the git branch name.

Adding the flatpak development remote (once):
```
flatpak --user remote-add --no-gpg-verify squey_dev http://inspector-cassiopee.ensam.eu/flatpak
```
Installing a development branch:
```
flatpak install --user squey_dev org.squey.Squey//<dev_branch_name>
```

### Roadmap

https://gitlab.com/groups/squey/-/roadmap
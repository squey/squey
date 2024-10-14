# About

[![Squey presentation video](https://gitlab.com/squey/squey/-/raw/main/buildstream/files/screenshots/squey_screenshot.png)](https://www.youtube.com/watch?v=M9I8Vt5mmAc)

See presentation video : https://www.youtube.com/watch?v=M9I8Vt5mmAc

## Project description

<!-- project_description_start -->

<p><a href="https://squey.org">Squey</a> is designed from the ground up to take advantage of GPUs and CPUs to perform interactive explorations of massive amounts of data.</p>

<p>It gives users an exhaustive yet intuitive multi-view representation of columnar data and can ingest from:</p>
<ol>
    <li>Structured text files (CSV, logs, ...)</li>
    <li>Apache Parquet files</li>
    <li>Pcap files</li>
    <li>SQL databases</li>
    <li>Elasticsearch databases</li>
</ol>

<p>Squey strives to deliver value through its <b>V.I.SU</b> approach:</p>

<ul>
  <li><b>Visualize</b>: Leverage various visual representations of raw data in combination with statistics</li>
  <li><b>Investigate</b>: Use filters to build an accurate understanding of millions of rows while switching instantly between capturing the big picture and focusing on the details</li>
  <li><b>Spot the Unknown</b>: As a structured understanding of the data emerges, identify unknowns and anomalies</li>
</ul>

<p>Squey can be used for many different purposes, such as:</p>
<ul>
    <li><b>BI and Big Data</b>: Bootstrap initial understanding of complex datasets and deep dive where necessary to design accurate data processing</li>
    <li><b>Cybersecurity</b>: Detect weak signals such as attacks and data leaks</li>
    <li><b>IT troubleshooting</b>: Resolve network issues and improve application performance</li>
    <li><b>Machine Learning</b>: Design training dataset to fulfill targeted improvements of Machine Learning models</li>
</ul>

<br>
<p>Give yourself a chance to <b>see</b> your data and have fun exploring!</p>

<!-- project_description_end -->

## Installation

### Linux

[<img src="https://flathub.org/assets/badges/flathub-badge-en.png" width="200"/>](https://flathub.org/apps/details/org.squey.Squey)

```
flatpak install --user -y https://dl.flathub.org/repo/appstream/org.squey.Squey.flatpakref
```

### Windows

Windows support is available through the use of [WSL2](https://apps.microsoft.com/store/detail/windows-subsystem-for-linux/9P9TQF7MRM4R):

[<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/Windows_logo_and_wordmark_-_2021.svg/320px-Windows_logo_and_wordmark_-_2021.svg.png" width="200"/>](https://squey.gitlab.io/squey/squey_installer.exe)


### AWS

Deploying the software as a service on an AWS EC2 instance :

[<img src="https://squey.org/images/logos/aws_marketplace.png" width="200"/>](https://aws.amazon.com/marketplace/search/results?searchTerms=Squey)


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

### Support the project

Use this DOI to make citations

[![Zenodo](https://zenodo.org/badge/648945293.svg)](https://doi.org/10.5281/zenodo.13927905)

Help the contributors to develop and maintain the software

[![LiberaPay](https://liberapay.com/assets/widgets/donate.svg)](https://liberapay.com/Squey/)
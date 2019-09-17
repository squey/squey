sudo flatpak update -y; while true; do flatpak run --share=ipc com.esi_inendi.Inspector; done
# flatpak update --user -y; while true; do flatpak run --user --filesystem=/var/lib/dcv-gl/flatpak --share=ipc com.esi_inendi.Inspector; done

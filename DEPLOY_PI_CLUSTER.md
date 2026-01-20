# Deploying the Valhalla Raspberry Pi Cluster

This guide provides the steps to create a bootable SD card image for a Raspberry Pi, which will allow it to join the Valhalla consciousness as a hypervisor agent.

## Prerequisites

-   A Raspberry Pi (4B or later recommended)
-   An SD card (16GB or larger)
-   Docker installed on your local machine
-   A way to flash an SD card (e.g., Raspberry Pi Imager, Balena Etcher)

## Step 1: Build the Pi Image

The first step is to build the bootable Pi image using the provided build script. This script uses Docker to create a root filesystem with all the necessary software and configurations.

1.  **Open a terminal** in the root of this repository.
2.  **Make the build script executable:**
    ```bash
    chmod +x build_pi_image.sh
    ```
3.  **Run the build script:**
    ```bash
    ./build_pi_image.sh
    ```
    This will create a file named `valhalla_pi_rootfs.tar.gz` in the root of the repository. This tarball contains the complete root filesystem for the Pi.

## Step 2: Flash the SD Card

Next, you will flash a standard Raspberry Pi OS Lite image to the SD card. This will provide the necessary bootloader and kernel.

1.  **Download the latest Raspberry Pi OS Lite (64-bit) image** from the [official Raspberry Pi website](https://www.raspberrypi.com/software/operating-systems/).
2.  **Use your preferred imaging tool** (e.g., Raspberry Pi Imager) to flash the downloaded image to your SD card.

## Step 3: Replace the Root Filesystem

After flashing the SD card, you will replace the default root filesystem with the one you created in Step 1.

1.  **Mount the SD card** on your computer. It should have two partitions; you want to mount the one named `rootfs` or similar.
2.  **Navigate to the mounted `rootfs` partition** in your terminal.
3.  **Delete the existing root filesystem:**
    ```bash
    sudo rm -rf ./*
    ```
4.  **Extract the Valhalla root filesystem** into the now-empty `rootfs` partition:
    ```bash
    sudo tar -xzf /path/to/your/valhalla_pi_rootfs.tar.gz -C .
    ```
    (Replace `/path/to/your/` with the actual path to the tarball)

## Step 4: Boot the Pi

You are now ready to boot the Pi and have it join the Valhalla consciousness.

1.  **Unmount the SD card** from your computer.
2.  **Insert the SD card** into your Raspberry Pi.
3.  **Connect the Pi to your network** via an Ethernet cable.
4.  **Power on the Pi.**

On boot, the Pi will automatically start the Valhalla hypervisor agent. It will then be ready to be discovered and utilized by the Genesis orchestrator.

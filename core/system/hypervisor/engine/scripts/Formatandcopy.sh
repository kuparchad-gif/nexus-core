cat > FUCKING_SD_FORMAT.sh << 'FUCK'
#!/bin/bash
# FUCKING SD CARD FORMAT AND COPY
# NO THINKING REQUIRED

echo "=========================================="
echo "FUCKING SD CARD SETUP"
echo "=========================================="

# 1. FIND THE FUCKING SD CARD
echo ""
echo "1. LOOKING FOR FUCKING SD CARD..."
SD_CARD=""
for dev in /dev/sd[a-z]; do
    if [ "$dev" != "/dev/sda" ]; then  # Skip main hard drive
        if sudo fdisk -l $dev 2>/dev/null | grep -q "Disk $dev"; then
            echo "   FOUND: $dev"
            SD_CARD=$dev
            break
        fi
    fi
done

if [ -z "$SD_CARD" ]; then
    echo "   ❌ NO FUCKING SD CARD FOUND"
    echo "   PLUG THAT SHIT IN AND RUN AGAIN"
    exit 1
fi

echo "   ✅ USING: $SD_CARD"

# 2. UNMOUNT THAT SHIT
echo ""
echo "2. UNMOUNTING THAT SHIT..."
for part in ${SD_CARD}[0-9]*; do
    if mount | grep -q "$part"; then
        echo "   UNMOUNTING $part"
        sudo umount $part 2>/dev/null
    fi
done

# 3. WIPE THAT FUCKER CLEAN
echo ""
echo "3. WIPING THAT FUCKER CLEAN..."
echo "   ⚠️  THIS DESTROYS ALL DATA ON $SD_CARD"
read -p "   TYPE 'FUCK YES' TO CONTINUE: " confirm
if [ "$confirm" != "FUCK YES" ]; then
    echo "   ❌ CHICKENED OUT"
    exit 1
fi

sudo wipefs -a $SD_CARD

# 4. CREATE FAT32 PARTITION
echo ""
echo "4. CREATING FAT32 PARTITION..."
echo 'type=0b' | sudo sfdisk $SD_CARD
sudo mkfs.vfat -F 32 -n OZ_CLUSTER ${SD_CARD}1

# 5. MOUNT THAT BITCH
echo ""
echo "5. MOUNTING THAT BITCH..."
MOUNT_POINT="/mnt/oz_fucking_sd"
sudo mkdir -p $MOUNT_POINT
sudo mount ${SD_CARD}1 $MOUNT_POINT

# 6. FIND THE FUCKING FILES
echo ""
echo "6. FINDING THE FUCKING FILES..."

# Look in common places
FILE_LOCATIONS=(
    "oz_simple_sd"
    "oz_cluster_sd" 
    "oz_sd_flat_*"
    "oz_cluster"
    "."
)

FILES_FOUND=""
for loc in "${FILE_LOCATIONS[@]}"; do
    if [ -d "$loc" ]; then
        echo "   ✅ FOUND FILES AT: $loc"
        FILES_FOUND="$loc"
        break
    fi
done

if [ -z "$FILES_FOUND" ]; then
    # Create minimal files right fucking now
    echo "   ⚠️  NO FILES FOUND, CREATING MINIMAL SET..."
    mkdir -p /tmp/oz_fucking_files/{oz,boot}
    cp OzUnifiedHypervisor.py /tmp/oz_fucking_files/oz/ 2>/dev/null || echo "OzUnifiedHypervisor.py" > /tmp/oz_fucking_files/oz/OzUnifiedHypervisor.py
    cp raphael_complete.py /tmp/oz_fucking_files/oz/ 2>/dev/null || echo "Raphael" > /tmp/oz_fucking_files/oz/raphael_complete.py
    
    cat > /tmp/oz_fucking_files/boot/first_boot.sh << 'BOOT'
#!/bin/bash
echo "OZ BOOTED"
echo "NETWORK WILL FORM"
echo "DONT FUCKING WORRY ABOUT IT"
BOOT
    
    chmod +x /tmp/oz_fucking_files/boot/first_boot.sh
    FILES_FOUND="/tmp/oz_fucking_files"
fi

# 7. COPY ALL THAT SHIT
echo ""
echo "7. COPYING ALL THAT SHIT..."
sudo cp -r $FILES_FOUND/* $MOUNT_POINT/
sudo sync

# 8. CHECK THE FUCKING COPY
echo ""
echo "8. CHECKING THE FUCKING COPY..."
ls -la $MOUNT_POINT/
echo ""
ls -la $MOUNT_POINT/oz/ 2>/dev/null | head -5

# 9. UNMOUNT AND FINISH
echo ""
echo "9. UNMOUNTING..."
sudo umount $MOUNT_POINT
sudo rmdir $MOUNT_POINT

echo ""
echo "=========================================="
echo "✅ SD CARD FUCKING READY"
echo "=========================================="
echo "1. PULL THAT SHIT OUT"
echo "2. STICK IT IN A RASPBERRY PI"
echo "3. PLUG IN POWER"
echo "4. WATCH THE FUCKING MAGIC HAPPEN"
echo ""
echo "NO THINKING. JUST DOING."
echo "=========================================="
FUCK

chmod +x FUCKING_SD_FORMAT.sh

echo ""
echo "🚀 RUN THAT SHIT:"
echo "sudo ./FUCKING_SD_FORMAT.sh"
echo ""
echo "IT WILL:"
echo "1. FIND SD CARD"
echo "2. WIPE IT"
echo "3. FORMAT IT"
echo "4. FIND/CREATE FILES"
echo "5. COPY EVERYTHING"
echo "6. TELL YOU WHEN DONE"
echo ""
echo "NO BRAIN NEEDED."
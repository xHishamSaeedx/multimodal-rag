# Docker Disk Cleanup Guide

## Problem

Docker Desktop on Windows uses WSL2 virtual disks (`.vhdx` files) that can grow but don't automatically shrink when you delete images or containers. This can lead to significant disk space being "wasted" even after cleaning up Docker resources.

**Symptoms:**
- Docker shows images using ~10-15 GB
- But Windows shows Docker using 40-50+ GB
- `docker system prune -a` reclaims 0B
- Disk space doesn't recover after deleting large images

## Root Cause

When you delete Docker images/containers, the space is freed **inside** the WSL2 virtual disk, but the `.vhdx` file itself doesn't shrink. The virtual disk file remains at its maximum size.

## Solution: Compact WSL2 Virtual Disk

### Prerequisites

- **Administrator privileges** (required for diskpart)
- Docker Desktop must be **completely closed**

### Step-by-Step Instructions

1. **Close Docker Desktop completely:**
   - Right-click Docker Desktop icon in system tray
   - Select "Quit Docker Desktop"
   - Wait a few seconds to ensure it's fully closed

2. **Open PowerShell as Administrator:**
   - Press `Win + X`
   - Select "Windows PowerShell (Admin)" or "Terminal (Admin)"

3. **Shutdown WSL2:**
   ```powershell
   wsl --shutdown
   ```

4. **Open diskpart:**
   ```powershell
   diskpart
   ```

5. **In diskpart, run these commands:**
   ```
   select vdisk file="C:\Users\<YOUR_USERNAME>\AppData\Local\Docker\wsl\disk\docker_data.vhdx"
   attach vdisk readonly
   compact vdisk
   detach vdisk
   exit
   ```

   **Note:** Replace `<YOUR_USERNAME>` with your actual Windows username, or use:
   ```
   select vdisk file="%LOCALAPPDATA%\Docker\wsl\disk\docker_data.vhdx"
   ```

6. **Wait for compaction to complete** (may take several minutes)

7. **Restart Docker Desktop**

### Verify Results

Check the file size before and after:
```powershell
Get-Item "$env:LOCALAPPDATA\Docker\wsl\disk\docker_data.vhdx" | 
    Select-Object Name, @{Name="Size(GB)";Expression={[math]::Round($_.Length/1GB,2)}}
```

## Alternative Methods

### Method 2: Docker Desktop Settings

Some versions of Docker Desktop have built-in cleanup options:

1. Open Docker Desktop
2. Go to **Settings → Resources → Advanced**
3. Look for "Clean / Purge data" or "Reclaim disk space" button
4. Some versions have a "Compact disk" option

### Method 3: Check Docker Disk Usage First

Before compacting, check what's actually using space:

```powershell
# Check Docker disk usage
docker system df -v

# Remove unused images (safe)
docker image prune -a

# Remove build cache (will slow down next build)
docker builder prune -a

# Comprehensive cleanup (removes unused everything)
docker system prune -a
```

**Note:** If these commands show "Total reclaimed space: 0B", then the space is in the WSL2 virtual disk and needs compaction.

## Expected Results

- **Before:** Virtual disk at 40-50+ GB
- **After:** Virtual disk shrinks to actual usage (~10-15 GB for typical setups)
- **Space recovered:** 30-40 GB typically

## Troubleshooting

### "The process cannot access the file because it is being used"

- Ensure Docker Desktop is **completely closed** (check Task Manager)
- Wait a few seconds after closing
- Try `wsl --shutdown` again

### "Access is denied" or diskpart fails

- Make sure PowerShell is running **as Administrator**
- Right-click PowerShell → "Run as Administrator"

### Compaction doesn't reduce size

- The virtual disk might actually be using that much space
- Check actual usage with `docker system df -v`
- Some space is reserved for Docker's internal operations

## Prevention

To prevent this issue in the future:

1. **Regular cleanup:**
   ```powershell
   docker system prune -a
   ```

2. **Set disk size limit in Docker Desktop:**
   - Settings → Resources → Advanced
   - Set "Disk image size" limit (e.g., 50 GB)

3. **Monitor disk usage:**
   ```powershell
   docker system df
   ```

## Related Files

- Docker WSL2 disk location: `%LOCALAPPDATA%\Docker\wsl\disk\docker_data.vhdx`
- Main WSL2 disk: `%LOCALAPPDATA%\Docker\wsl\main\ext4.vhdx` (usually small)

## References

- [Docker Desktop Disk Usage](https://docs.docker.com/desktop/troubleshoot/troubleshoot/#disk-usage)
- [WSL2 Virtual Disk Compaction](https://learn.microsoft.com/en-us/windows/wsl/disk-space)


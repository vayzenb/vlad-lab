# Essential Linux Admin Commands - Quick Reference

## User and group Management
```bash
# Show current user info
whoami
id $USER

# List users
ls /home


# Add user to group
sudo usermod -aG group username

# Remove user from group
sudo gpasswd -d username group

# List all groups
cat /etc/group

# Show groups for specific user
groups [username]

# Show group info
getent group [groupname]

#reset your terminal after being added to a group
exec bash
```


## Package Management (Ubuntu/Debian)
```bash
# Update packages
sudo apt update
sudo apt upgrade

# Install/remove
sudo apt install package_name
sudo apt remove package_name

# Search
apt search keyword
```


## File Permissions
```bash

# view all files and their permissions
ls -l 

# Change permissions

#common permission codes are: granted (r), write (w), execute (x)
chmod 755 file
chmod g+rw file

# add group permissions to directory recusrively
chmod -R g+rwx /path/to/folder

# Change ownership
sudo chown user:group file
sudo chown -R user:group directory
```

## System Info
```bash
# System resources
df -h                    # Disk space
free -h                  # Memory
top                      # CPU/memory usage
uptime                   # System load


```

## ZFS Essentials


### Snapshots
```bash
# List datasets
zfs list

# List snapshots
zfs list -t snapshot

# Delete snapshot
sudo zfs destroy tank/data@backup-20250101

# Rollback
sudo zfs rollback tank/data@backup-20250101
```


# Common Bash Commands Reference

A  guide to the most commonly used bash commands with examples and explanations.

## File and Directory Operations

### Navigation Commands

```bash
# Print current working directory
pwd

# List files and directories
ls                    # Basic listing

# Change directory
cd /path/to/directory # Go to specific directory
cd ..                 # Go up one directory level

```

## Running jobs

Let's say you want to run multiple analyses at once or you have an analysis that needs to run for 3 days (!!). You don't want this analysis to get interrupted if you accidentally close your window. This is where the `screen` command comes in.

Think of screens as seperate workplaces that can be opened, used to runn jobs, and thin minimized without interrupting the job.

```bash
# Open a new screen with
screen -R screen_name #screen name can be anything (e.g., screen -R fmri_analysis)

# Run some code in the new screen then minimize it by press Ctrl+A then D
# To repoen the screen simply type screen -R with the name of the screen

# To close a screen from inside it, simply type
exit

#Sometimes you might want to kill a screen from outside of it
screen -XS screen_name quit

#You can have as many screen open as you want
screen -ls #list all the screens you currently have

#make sure to clsoe these intermittently


```


### File Operations

```bash

# Copy files and directories
cp source.txt destination.txt        # Copy file
cp -r source_dir/ destination_dir/   # Copy directory recursively (i.e., across subfolders)

# Move/rename files and directories
mv oldname.txt newname.txt          # Rename file
mv file.txt /path/to/directory/     # Move file to directory

# Remove files and directories
rm filename.txt                     # Remove file
rm -rf directory/                   # Force remove directory (be careful!)

```

### Directory Operations

```bash
# Create directories
mkdir new_directory                 # Create single directory

# Find files and directories
find /path/to/search -name "*.txt"     # Find files by name pattern
find . -type f -name "*.log"           # Find files with .log extension
find . -type d -name "project*"        # Find directories starting with "project"
find . -size +100M                     # Find files larger than 100MB
find . -mtime -7                       # Find files modified in last 7 days
```

### Managing Environmnent Variables

```bash

# create a new 

### Environment Variables

```bash
# View environment variables
env                                # Show all environment variables
echo $VARIABLE                         # Display specific variable

```

## File Transfer


## Tips for Effective Bash Usage

1. **Use Tab completion** - Press Tab to auto-complete commands and file names
2. **Use wildcards** - `*` matches any characters, `?` matches single character
3. **Chain commands** - Use `&&` for conditional execution, `;` for sequential execution
4. Run code in screens!

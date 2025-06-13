## GitHub set up

### In a browser
1. In a seperate window go to: GitHub.com → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token"
3. Set Expiration to 'No Expiration'
4. Select scopes: I would select all of them
5. Copy the token and save it somewhere(You won't see it again)

### In a terminal connected to ARTOO

Configure Git:
```bash
# Set your username and email
git config --global user.name "Your Username"
git config --global user.email "your-email@example.com"

# Prompt for login info
gh auth login

# Paste the token when prompted
```

## Basic Commands

```bash
# Cloning a repository
git clone https://github.com/vayzenb/vlad-lab.git

# When inside a repository folder and committing some changes
git add -A
git commit -m "some message about what you did"
git push

# When pulling new changes from a repository
git pull
```
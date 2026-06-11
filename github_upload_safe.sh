#!/usr/bin/env bash
# Safe GitHub upload helper for Cursor/one-shot terminals.
# It never exits immediately on command failure; errors are printed and the shell stays open.

set +e

ROOT="${HOME}/ur3_control"
cd "${ROOT}" || {
  echo "ERROR: cannot cd to ${ROOT}"
  exec "${SHELL:-/bin/bash}" -i
}

echo "Current folder: $(pwd)"
echo ""

if [ ! -d .git ]; then
  echo "Initializing git repository..."
  git init
fi

if [ ! -f .gitignore ]; then
  cat > .gitignore <<'EOF'
build/
install/
log/
__pycache__/
*.pyc
.pytest_cache/
*.bag
*.db3
*.pt
*.onnx
*.engine
my_ur3_calibration.yaml
.env
*.secret
EOF
fi

echo "Git status:"
git status --short --branch
echo ""

current_remote="$(git remote get-url origin 2>/dev/null)"
if [ -z "${current_remote}" ] || [ "${current_remote}" = "YOUR_GITHUB_REPO_URL" ]; then
  echo "Enter your real GitHub repository URL."
  echo "https://github.com/wonny-eun/ROS2_MJC_UR3_new/wonny-eun/ur3_control.git"
  read -r -p "GitHub repo URL: " repo_url
  if [ -z "${repo_url}" ]; then
    echo "ERROR: no repo URL entered. Nothing pushed."
    exec "${SHELL:-/bin/bash}" -i
  fi
  git remote remove origin 2>/dev/null
  git remote add origin "${repo_url}"
fi

if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "Creating initial commit..."
  git add .
  git commit -m "Initial UR3 control workspace"
else
  echo "Repository already has a commit:"
  git log --oneline -1
fi

echo ""
echo "Remote:"
git remote -v
echo ""

echo "Pushing main branch..."
git branch -M main
git push -u origin main
push_code=$?

echo ""
if [ "${push_code}" -eq 0 ]; then
  echo "GitHub upload finished successfully."
else
  echo "GitHub upload failed with exit code ${push_code}."
  echo "Read the error above. Common causes: wrong URL, not logged in, private repo permission, or large files."
fi

echo ""
echo "Shell stays open. Type commands here, or type exit to close."
exec "${SHELL:-/bin/bash}" -i

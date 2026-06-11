#!/usr/bin/env bash
# Push ~/ur3_control to the existing GitHub repo safely.
# It uses a new branch so the existing GitHub main branch is not overwritten.

set +e

ROOT="${HOME}/ur3_control"
REMOTE_URL="https://github.com/wonny-eun/ROS2_MJC_UR3_new.git"
BRANCH_NAME="ur3-control-upload"

cd "${ROOT}" || {
  echo "ERROR: cannot cd to ${ROOT}"
  exec "${SHELL:-/bin/bash}" -i
}

echo "Repository: $(pwd)"
echo "Remote: ${REMOTE_URL}"
echo "Upload branch: ${BRANCH_NAME}"
echo ""

if [ ! -d .git ]; then
  echo "ERROR: ${ROOT} is not a git repository."
  echo "Run git init first."
  exec "${SHELL:-/bin/bash}" -i
fi

git remote remove origin 2>/dev/null
git remote add origin "${REMOTE_URL}"

echo "Checking GitHub login..."
gh auth status >/dev/null 2>&1
if [ "$?" -ne 0 ]; then
  echo "You are not logged into GitHub."
  echo "A browser login will start now. Finish it, then this script will continue."
  gh auth login --hostname github.com --web
  if [ "$?" -ne 0 ]; then
    echo "ERROR: GitHub login failed."
    exec "${SHELL:-/bin/bash}" -i
  fi
  gh auth setup-git 2>/dev/null
fi

echo ""
echo "Current git status:"
git status --short --branch
echo ""

if ! git rev-parse --verify HEAD >/dev/null 2>&1; then
  echo "No commit exists. Creating initial commit."
  git add .
  git commit -m "Initial UR3 control workspace"
fi

echo "Pushing to ${REMOTE_URL} branch ${BRANCH_NAME}..."
git push -u origin "HEAD:${BRANCH_NAME}"
push_code=$?

echo ""
if [ "${push_code}" -eq 0 ]; then
  echo "Upload succeeded."
  echo "Open: https://github.com/wonny-eun/ROS2_MJC_UR3_new/tree/${BRANCH_NAME}"
  echo "You can later merge this branch into main on GitHub."
else
  echo "Upload failed with exit code ${push_code}."
  echo "Read the error above. The shell will stay open."
fi

exec "${SHELL:-/bin/bash}" -i

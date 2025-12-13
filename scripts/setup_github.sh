#!/bin/bash
# Script para configurar o remote GitHub após criar o repositório manualmente
# Uso: ./scripts/setup_github.sh <github-username> <repo-name>

if [ $# -ne 2 ]; then
    echo "Uso: $0 <github-username> <repo-name>"
    echo "Exemplo: $0 martimnicolau prodplan4"
    exit 1
fi

GITHUB_USER=$1
REPO_NAME=$2

echo "Configurando remote GitHub..."
echo "Repositório: https://github.com/${GITHUB_USER}/${REPO_NAME}"

# Adicionar remote origin
git remote add origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git" 2>/dev/null || \
git remote set-url origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

# Verificar remote
echo ""
echo "Remote configurado:"
git remote -v

echo ""
echo "Para fazer push, execute:"
echo "  git push -u origin main"
echo ""
echo "Nota: Pode ser necessário autenticar. Use:"
echo "  - Personal Access Token (recomendado)"
echo "  - SSH key (se configurado)"
echo "  - GitHub CLI: gh auth login"


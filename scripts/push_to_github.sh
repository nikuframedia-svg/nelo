#!/bin/bash
# Script para fazer push para GitHub
# Uso: ./scripts/push_to_github.sh <github-username> <repo-name>
# Exemplo: ./scripts/push_to_github.sh martimnicolau prodplan4

if [ $# -ne 2 ]; then
    echo "❌ Uso: $0 <github-username> <repo-name>"
    echo "Exemplo: $0 martimnicolau prodplan4"
    exit 1
fi

GITHUB_USER=$1
REPO_NAME=$2

echo "🔧 Configurando remote GitHub..."
echo "Repositório: https://github.com/${GITHUB_USER}/${REPO_NAME}"

# Verificar se remote já existe
if git remote get-url origin >/dev/null 2>&1; then
    echo "⚠️  Remote 'origin' já existe. Atualizando..."
    git remote set-url origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
else
    echo "➕ Adicionando remote 'origin'..."
    git remote add origin "https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
fi

# Verificar remote
echo ""
echo "✅ Remote configurado:"
git remote -v

echo ""
echo "📤 Fazendo push para GitHub..."
echo "Branch: main"
echo ""

# Tentar push
if git push -u origin main; then
    echo ""
    echo "✅ Push realizado com sucesso!"
    echo "🌐 Repositório: https://github.com/${GITHUB_USER}/${REPO_NAME}"
else
    echo ""
    echo "❌ Erro ao fazer push."
    echo ""
    echo "Possíveis causas:"
    echo "1. Repositório não existe no GitHub - crie em: https://github.com/new"
    echo "2. Problemas de autenticação - use Personal Access Token"
    echo "3. Branch remota diferente - verifique no GitHub"
    echo ""
    echo "Para autenticar:"
    echo "  - Use Personal Access Token como password"
    echo "  - Ou configure SSH: git remote set-url origin git@github.com:${GITHUB_USER}/${REPO_NAME}.git"
    exit 1
fi


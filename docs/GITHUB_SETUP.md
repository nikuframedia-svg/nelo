# Configuração do Repositório GitHub

## Status Atual

✅ **Git inicializado** no diretório local  
✅ **Commit inicial criado** (242 ficheiros, 105484 linhas)  
✅ **Branch principal**: `main`  
⏳ **Aguardando**: Criação do repositório no GitHub

## Próximos Passos

### Opção 1: Criar Repositório via Interface Web (Recomendado)

1. **Aceder ao GitHub**: https://github.com/new

2. **Criar novo repositório**:
   - **Nome**: `prodplan4` (ou outro nome de sua escolha)
   - **Visibilidade**: Private (recomendado) ou Public
   - **NÃO** inicializar com README, .gitignore ou LICENSE (já temos)

3. **Configurar remote e fazer push**:

   ```bash
   # Usar o script helper
   ./scripts/setup_github.sh <seu-username> <nome-do-repo>
   
   # Exemplo:
   ./scripts/setup_github.sh martimnicolau prodplan4
   
   # Ou manualmente:
   git remote add origin https://github.com/<seu-username>/<nome-do-repo>.git
   git push -u origin main
   ```

### Opção 2: Criar Repositório via GitHub CLI

Se tiver GitHub CLI instalado:

```bash
# Autenticar (se necessário)
gh auth login

# Criar repositório
gh repo create prodplan4 --private --source=. --remote=origin --push
```

### Opção 3: Criar Repositório via API

```bash
# Usar token de Personal Access Token (armazenado em variável de ambiente)
export GITHUB_TOKEN=ghp_...

curl -X POST \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/user/repos \
  -d '{"name":"prodplan4","private":true}'

# Depois configurar remote
git remote add origin https://github.com/<seu-username>/prodplan4.git
git push -u origin main
```

## Autenticação

Para fazer push, pode ser necessário autenticar:

### Personal Access Token (Recomendado)

1. Criar token: https://github.com/settings/tokens
2. Permissões: `repo` (acesso completo a repositórios privados)
3. Usar token como password ao fazer push

### SSH Key

Se preferir usar SSH:

```bash
# Configurar remote SSH
git remote set-url origin git@github.com:<seu-username>/<nome-do-repo>.git

# Fazer push
git push -u origin main
```

## Verificação

Após configurar o remote:

```bash
# Verificar remote
git remote -v

# Deve mostrar:
# origin  https://github.com/<seu-username>/<nome-do-repo>.git (fetch)
# origin  https://github.com/<seu-username>/<nome-do-repo>.git (push)
```

## Segurança

⚠️ **IMPORTANTE**: 
- NUNCA commitar tokens, chaves ou credenciais
- Verificar `.gitignore` antes de cada commit
- Usar variáveis de ambiente para configuração sensível

## Estrutura do Repositório

O repositório contém:

```
.
├── backend/              # Backend FastAPI
├── factory-optimizer/     # Frontend React
├── data/                 # Dados de exemplo
├── docs/                 # Documentação
├── scripts/              # Scripts auxiliares
├── .gitignore            # Regras de exclusão
└── README.md             # Documentação principal
```

---

**Próximo passo**: Criar o repositório no GitHub e executar `./scripts/setup_github.sh <username> <repo-name>`


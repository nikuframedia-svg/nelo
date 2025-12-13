"""
API para Consultas Técnicas - Backend determinístico

Endpoints para perguntas técnicas sobre o modelo APS.
O LLM pode chamar estes endpoints e reformular as respostas.

NUNCA o LLM inventa respostas - sempre consulta estes endpoints.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.aps.technical_queries import get_technical_queries

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/alternatives")
async def get_alternatives(
    artigo: str = Query(..., description="Artigo (ex: 'GO Artigo 6')"),
    rota: str = Query(..., description="Rota (ex: 'Rota A')"),
    operacao: str = Query(..., description="ID da operação (ex: 'OP-001')"),
):
    """
    Retorna alternativas de máquinas para uma operação específica.
    
    Resposta determinística baseada no Excel parseado.
    """
    try:
        tech_queries = get_technical_queries()
        alternatives = tech_queries.get_alternatives(artigo, rota, operacao)
        
        if not alternatives:
            raise HTTPException(
                status_code=404,
                detail=f"Nenhuma alternativa encontrada para {artigo}/{rota}/{operacao}"
            )
        
        return {
            "artigo": artigo,
            "rota": rota,
            "operacao": operacao,
            "alternatives": alternatives,
        }
    except Exception as exc:
        logger.exception(f"Erro ao obter alternativas: {exc}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter alternativas: {str(exc)}")


@router.get("/routes")
async def get_routes(
    artigo: str = Query(..., description="Artigo (ex: 'GO Artigo 6')"),
):
    """
    Retorna rotas disponíveis para um artigo.
    
    Resposta determinística baseada no Excel parseado.
    """
    try:
        tech_queries = get_technical_queries()
        routes = tech_queries.get_routes(artigo)
        
        if not routes:
            raise HTTPException(
                status_code=404,
                detail=f"Nenhuma rota encontrada para artigo '{artigo}'"
            )
        
        return {
            "artigo": artigo,
            "routes": routes,
        }
    except Exception as exc:
        logger.exception(f"Erro ao obter rotas: {exc}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter rotas: {str(exc)}")


@router.get("/operations")
async def get_operations_on_machine(
    machine_id: str = Query(..., description="ID da máquina (ex: '300')"),
):
    """
    Retorna operações que podem ser feitas numa máquina.
    
    Resposta determinística baseada no Excel parseado.
    """
    try:
        tech_queries = get_technical_queries()
        operations = tech_queries.get_operations_on_machine(machine_id)
        
        return {
            "machine_id": machine_id,
            "operations": operations,
        }
    except Exception as exc:
        logger.exception(f"Erro ao obter operações: {exc}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter operações: {str(exc)}")


@router.get("/families")
async def get_families_on_machine(
    machine_id: str = Query(..., description="ID da máquina (ex: '300')"),
):
    """
    Retorna famílias que usam uma máquina.
    
    Resposta determinística baseada no Excel parseado.
    """
    try:
        tech_queries = get_technical_queries()
        families = tech_queries.get_families_on_machine(machine_id)
        
        return {
            "machine_id": machine_id,
            "families": families,
        }
    except Exception as exc:
        logger.exception(f"Erro ao obter famílias: {exc}")
        raise HTTPException(status_code=500, detail=f"Erro ao obter famílias: {str(exc)}")


@router.get("/validate")
async def validate_entity(
    entity_type: str = Query(..., description="Tipo: 'machine', 'article', 'route', 'operation'"),
    machine_id: Optional[str] = Query(None, description="ID da máquina"),
    artigo: Optional[str] = Query(None, description="Artigo"),
    rota: Optional[str] = Query(None, description="Rota"),
    operacao: Optional[str] = Query(None, description="Operação"),
):
    """
    Valida se uma entidade existe no modelo.
    
    Resposta determinística baseada no Excel parseado.
    """
    try:
        tech_queries = get_technical_queries()
        
        if entity_type == "machine":
            if not machine_id:
                raise HTTPException(status_code=400, detail="machine_id é obrigatório")
            exists = tech_queries.validate_machine(machine_id)
            return {"entity_type": "machine", "entity_id": machine_id, "exists": exists}
        
        elif entity_type == "article":
            if not artigo:
                raise HTTPException(status_code=400, detail="artigo é obrigatório")
            exists = tech_queries.validate_article(artigo)
            return {"entity_type": "article", "entity_id": artigo, "exists": exists}
        
        elif entity_type == "route":
            if not artigo or not rota:
                raise HTTPException(status_code=400, detail="artigo e rota são obrigatórios")
            exists = tech_queries.validate_route(artigo, rota)
            return {"entity_type": "route", "artigo": artigo, "rota": rota, "exists": exists}
        
        elif entity_type == "operation":
            if not artigo or not rota or not operacao:
                raise HTTPException(status_code=400, detail="artigo, rota e operacao são obrigatórios")
            exists = tech_queries.validate_operation(artigo, rota, operacao)
            return {
                "entity_type": "operation",
                "artigo": artigo,
                "rota": rota,
                "operacao": operacao,
                "exists": exists,
            }
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Tipo de entidade inválido: {entity_type}. Válidos: machine, article, route, operation"
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(f"Erro ao validar entidade: {exc}")
        raise HTTPException(status_code=500, detail=f"Erro ao validar entidade: {str(exc)}")


import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile
from fastapi import Request
import traceback
from pydantic import BaseModel, Field

from app.etl.loader import get_loader


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

router = APIRouter(prefix="/etl", tags=["ETL"])


class MappingUpdateRequest(BaseModel):
    file: str = Field(..., description="Nome do ficheiro carregado (ex.: Nikufra DadosProducao.xlsx)")
    sheet: str = Field(..., description="Nome da folha dentro do Excel")
    confirm: Dict[str, str] = Field(default_factory=dict, description="Mapeamentos confirmados {coluna_origem: coluna_canónica}")
    global_overrides: Optional[Dict[str, str]] = Field(
        default=None,
        description="Mapeamentos globais aplicáveis a qualquer folha",
    )


def _normalize_uploads(payload: Optional[Union[UploadFile, List[UploadFile]]]) -> List[UploadFile]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return [item for item in payload if item is not None]
    return [payload]


@router.post("/upload")
async def etl_upload(
    request: Request,
    files: Optional[Union[UploadFile, List[UploadFile]]] = File(None, description="Ficheiro(s) Excel"),
    file: Optional[Union[UploadFile, List[UploadFile]]] = File(None, description="Ficheiro(s) Excel (alias)"),
):
    bucket: List[UploadFile] = []
    bucket.extend(_normalize_uploads(files))
    bucket.extend(_normalize_uploads(file))

    form = await request.form()
    for key in ("files", "file"):
        for item in form.getlist(key):
            if hasattr(item, "filename") and item not in bucket:
                bucket.append(item)

    print("UPLOAD PARTS:", [getattr(upload, "filename", None) for upload in bucket])
    logger.info("UPLOAD PARTS: %s", [getattr(upload, "filename", None) for upload in bucket])

    if not bucket:
        raise HTTPException(status_code=400, detail="Nenhum ficheiro enviado.")

    loader = get_loader()
    batch_id = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    versions_dir = loader.data_dir / "versions" / batch_id
    versions_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for upload in bucket:
        filename = Path(getattr(upload, "filename", "") or "").name
        if not filename:
            continue
        if not filename.lower().endswith(".xlsx"):
            raise HTTPException(status_code=400, detail=f"Formato inválido: {filename}")

        content = await upload.read()
        await upload.close()
        if not content:
            continue

        version_path = versions_dir / filename
        version_path.write_bytes(content)

        target_path = loader.data_dir / filename
        target_path.write_bytes(content)
        saved_paths.append(target_path)

    if not saved_paths:
        raise HTTPException(status_code=400, detail="Nenhum ficheiro válido para processar.")

    try:
        loader.process_uploaded_files(saved_paths)
    except Exception as exc:  # pylint: disable=broad-except
        trace = traceback.format_exc()
        logger.exception("Erro ao processar upload ETL: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc), "trace": trace}) from exc

    if hasattr(loader, "reload"):
        try:
            reload_callable = getattr(loader, "reload")
            if callable(reload_callable):
                try:
                    reload_callable(batch_id)  # type: ignore[misc]
                except TypeError:
                    reload_callable(batch_id=batch_id)  # type: ignore[misc]
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("loader.reload falhou para batch %s: %s", batch_id, exc)

    # Guardar batch_id no status
    status = loader.get_status()
    status["latest_batch_id"] = batch_id
    status["batch_id"] = batch_id
    loader._save_status_to_disk()  # type: ignore
    
    # Invalidar cache de insights do batch anterior (se existir)
    from app.insights.cache import get_insight_cache
    cache = get_insight_cache()
    old_batch_id = status.get("batch_id")
    if old_batch_id and old_batch_id != batch_id:
        # Invalidar todos os modes do batch anterior
        for mode in ["planeamento", "gargalos", "inventario", "resumo", "sugestoes"]:
            cache.invalidate(old_batch_id, mode)
    
    # NÃO chamar LLM aqui - apenas ETL + ML
    # O LLM será chamado on-demand quando o utilizador aceder a uma página
    
    return {"ok": True, "count": len(saved_paths), "batch_id": batch_id, "status": status}


@router.get("/preview")
async def preview_file(
    file: str = Query(..., description="Nome do ficheiro armazenado em app/data"),
    sheet: Optional[str] = Query(None, description="Nome da folha (opcional)"),
    limit: int = Query(5, ge=1, le=50, description="Número de linhas em preview"),
):
    loader = get_loader()
    try:
        return loader.generate_preview(file, sheet=sheet, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        trace = traceback.format_exc()
        logger.exception("Erro ao gerar preview ETL: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc), "trace": trace}) from exc


@router.post("/mapping")
async def update_mapping(payload: MappingUpdateRequest = Body(...)):
    loader = get_loader()
    try:
        result = loader.update_mappings(
            file_name=payload.file,
            sheet_name=payload.sheet,
            overrides=payload.confirm,
            global_overrides=payload.global_overrides,
        )
        return {"ok": True, "mapping": result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        trace = traceback.format_exc()
        logger.exception("Erro ao atualizar mappings ETL: %s", exc)
        raise HTTPException(status_code=500, detail={"error": str(exc), "trace": trace}) from exc


@router.get("/status")
async def etl_status():
    """Retorna status do ETL com tratamento de erros robusto."""
    try:
        loader = get_loader()
        status = loader.get_status()
        # Garantir que o status é serializável
        if not isinstance(status, dict):
            logger.warning(f"Status não é um dict: {type(status)}")
            return {"error": "Status inválido", "ready": False}
        return status
    except Exception as exc:
        logger.exception(f"Erro ao obter status do ETL: {exc}")
        # Retornar status mínimo em caso de erro
        return {
            "ready": False,
            "error": str(exc),
            "planning_ready": False,
            "inventory_ready": False,
            "has_data": False,
        }


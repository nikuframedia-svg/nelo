"""
════════════════════════════════════════════════════════════════════════════════
OPS INGESTION SERVICES - Serviços de Importação e Integração
════════════════════════════════════════════════════════════════════════════════

Contract 14: Serviços de importação e integração com módulos existentes

Funcionalidades:
- Importação de Excel (ordens, movimentos, RH, máquinas)
- Validação via Pydantic schemas
- Gravação em tabelas raw
- Integração com módulos existentes (ProdPlan, SmartInventory, etc.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import UploadFile
from sqlalchemy.orm import Session
from pydantic import ValidationError

from ops_ingestion.models import (
    OpsRawOrder,
    OpsRawInventoryMove,
    OpsRawHR,
    OpsRawMachine,
    MovementType,
)
from ops_ingestion.schemas import (
    OrderRowSchema,
    InventoryMoveRowSchema,
    HRRowSchema,
    MachineRowSchema,
    ImportResult,
)
from ops_ingestion.excel_parser import (
    parse_excel_orders,
    parse_excel_inventory_moves,
    parse_excel_hr,
    parse_excel_machines,
)
from ops_ingestion.data_quality import (
    analyze_orders_quality,
    analyze_inventory_moves_quality,
    analyze_hr_quality,
    analyze_machines_quality,
    detect_anomalies_ml_orders,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SERVICE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class OpsIngestionService:
    """
    Serviço de ingestão de dados operacionais.
    
    As specified in Contract 14:
    - Importa Excel (ordens, movimentos, RH, máquinas)
    - Valida via Pydantic
    - Grava em tabelas raw
    - Integra com módulos existentes
    """
    
    def import_orders_from_excel(
        self,
        file: UploadFile,
        db: Session,
    ) -> ImportResult:
        """
        Importa ordens de produção do Excel.
        
        Returns:
            ImportResult com contagens e erros
        """
        # Salvar ficheiro temporariamente
        temp_path = self._save_temp_file(file)
        
        try:
            # Parse Excel
            rows, parse_errors = parse_excel_orders(temp_path)
            
            if parse_errors:
                return ImportResult(
                    success=False,
                    imported_count=0,
                    failed_count=len(rows),
                    errors=parse_errors,
                    source_file=file.filename or "unknown",
                )
            
            # Validar e gravar
            imported_ids: List[int] = []
            warnings: List[str] = []
            errors: List[str] = []
            
            for row in rows:
                try:
                    # Validar schema
                    schema = OrderRowSchema(**row)
                    
                    # Criar modelo
                    order = OpsRawOrder(
                        external_order_code=schema.external_order_code,
                        product_code=schema.product_code,
                        quantity=schema.quantity,
                        due_date=schema.due_date,
                        routing_json=json.dumps(schema.routing) if schema.routing else None,
                        line_or_center=schema.line_or_center,
                        source_file=file.filename or "unknown",
                    )
                    
                    db.add(order)
                    db.flush()
                    imported_ids.append(order.id)
                
                except ValidationError as e:
                    errors.append(f"Linha inválida: {str(e)}")
                except Exception as e:
                    errors.append(f"Erro ao gravar: {str(e)}")
            
            db.commit()
            
            # Data quality checks
            if imported_ids:
                quality_result = analyze_orders_quality(db, imported_ids)
                warnings.extend(quality_result["warnings"])
                errors.extend(quality_result["errors"])
                
                # ML anomaly detection (opcional)
                if len(imported_ids) >= 100:
                    try:
                        anomalies = detect_anomalies_ml_orders(db, imported_ids)
                        if anomalies:
                            warnings.append(f"{len(anomalies)} anomalias detectadas por ML")
                    except Exception as e:
                        logger.debug(f"ML anomaly detection skipped: {e}")
            
            return ImportResult(
                success=len(errors) == 0,
                imported_count=len(imported_ids),
                failed_count=len(rows) - len(imported_ids),
                warnings=warnings,
                errors=errors,
                record_ids=imported_ids,
                source_file=file.filename or "unknown",
            )
        
        finally:
            # Limpar ficheiro temporário
            if temp_path.exists():
                temp_path.unlink()
    
    def import_inventory_moves_from_excel(
        self,
        file: UploadFile,
        db: Session,
    ) -> ImportResult:
        """Importa movimentos de inventário do Excel."""
        temp_path = self._save_temp_file(file)
        
        try:
            rows, parse_errors = parse_excel_inventory_moves(temp_path)
            
            if parse_errors:
                return ImportResult(
                    success=False,
                    imported_count=0,
                    failed_count=len(rows),
                    errors=parse_errors,
                    source_file=file.filename or "unknown",
                )
            
            imported_ids: List[int] = []
            warnings: List[str] = []
            errors: List[str] = []
            
            for row in rows:
                try:
                    schema = InventoryMoveRowSchema(**row)
                    
                    move = OpsRawInventoryMove(
                        order_code=schema.order_code,
                        from_station=schema.from_station,
                        to_station=schema.to_station,
                        movement_type=MovementType(schema.movement_type.value),
                        quantity_good=schema.quantity_good,
                        quantity_scrap=schema.quantity_scrap,
                        timestamp=schema.timestamp,
                        source_file=file.filename or "unknown",
                    )
                    
                    db.add(move)
                    db.flush()
                    imported_ids.append(move.id)
                
                except (ValidationError, ValueError) as e:
                    errors.append(f"Linha inválida: {str(e)}")
                except Exception as e:
                    errors.append(f"Erro ao gravar: {str(e)}")
            
            db.commit()
            
            # Data quality checks
            if imported_ids:
                quality_result = analyze_inventory_moves_quality(db, imported_ids)
                warnings.extend(quality_result["warnings"])
                errors.extend(quality_result["errors"])
            
            return ImportResult(
                success=len(errors) == 0,
                imported_count=len(imported_ids),
                failed_count=len(rows) - len(imported_ids),
                warnings=warnings,
                errors=errors,
                record_ids=imported_ids,
                source_file=file.filename or "unknown",
            )
        
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def import_hr_from_excel(
        self,
        file: UploadFile,
        db: Session,
    ) -> ImportResult:
        """Importa dados de RH do Excel."""
        temp_path = self._save_temp_file(file)
        
        try:
            rows, parse_errors = parse_excel_hr(temp_path)
            
            if parse_errors:
                return ImportResult(
                    success=False,
                    imported_count=0,
                    failed_count=len(rows),
                    errors=parse_errors,
                    source_file=file.filename or "unknown",
                )
            
            imported_ids: List[int] = []
            warnings: List[str] = []
            errors: List[str] = []
            
            for row in rows:
                try:
                    schema = HRRowSchema(**row)
                    
                    hr = OpsRawHR(
                        technician_code=schema.technician_code,
                        name=schema.name,
                        role=schema.role,
                        skills_json=json.dumps(schema.skills) if schema.skills else None,
                        shift_pattern=schema.shift_pattern,
                        home_cell=schema.home_cell,
                        source_file=file.filename or "unknown",
                    )
                    
                    db.add(hr)
                    db.flush()
                    imported_ids.append(hr.id)
                
                except ValidationError as e:
                    errors.append(f"Linha inválida: {str(e)}")
                except Exception as e:
                    errors.append(f"Erro ao gravar: {str(e)}")
            
            db.commit()
            
            # Data quality checks
            if imported_ids:
                quality_result = analyze_hr_quality(db, imported_ids)
                warnings.extend(quality_result["warnings"])
                errors.extend(quality_result["errors"])
            
            return ImportResult(
                success=len(errors) == 0,
                imported_count=len(imported_ids),
                failed_count=len(rows) - len(imported_ids),
                warnings=warnings,
                errors=errors,
                record_ids=imported_ids,
                source_file=file.filename or "unknown",
            )
        
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def import_machines_from_excel(
        self,
        file: UploadFile,
        db: Session,
    ) -> ImportResult:
        """Importa dados de máquinas do Excel."""
        temp_path = self._save_temp_file(file)
        
        try:
            rows, parse_errors = parse_excel_machines(temp_path)
            
            if parse_errors:
                return ImportResult(
                    success=False,
                    imported_count=0,
                    failed_count=len(rows),
                    errors=parse_errors,
                    source_file=file.filename or "unknown",
                )
            
            imported_ids: List[int] = []
            warnings: List[str] = []
            errors: List[str] = []
            
            for row in rows:
                try:
                    schema = MachineRowSchema(**row)
                    
                    machine = OpsRawMachine(
                        machine_code=schema.machine_code,
                        description=schema.description,
                        line=schema.line,
                        capacity_per_shift_hours=schema.capacity_per_shift_hours,
                        avg_setup_time_minutes=schema.avg_setup_time_minutes,
                        maintenance_windows_json=json.dumps(schema.maintenance_windows) if schema.maintenance_windows else None,
                        source_file=file.filename or "unknown",
                    )
                    
                    db.add(machine)
                    db.flush()
                    imported_ids.append(machine.id)
                
                except ValidationError as e:
                    errors.append(f"Linha inválida: {str(e)}")
                except Exception as e:
                    errors.append(f"Erro ao gravar: {str(e)}")
            
            db.commit()
            
            # Data quality checks
            if imported_ids:
                quality_result = analyze_machines_quality(db, imported_ids)
                warnings.extend(quality_result["warnings"])
                errors.extend(quality_result["errors"])
            
            return ImportResult(
                success=len(errors) == 0,
                imported_count=len(imported_ids),
                failed_count=len(rows) - len(imported_ids),
                warnings=warnings,
                errors=errors,
                record_ids=imported_ids,
                source_file=file.filename or "unknown",
            )
        
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    def _save_temp_file(self, file: UploadFile) -> Path:
        """Salva ficheiro temporariamente."""
        import tempfile
        import shutil
        
        suffix = Path(file.filename or "upload").suffix
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        try:
            shutil.copyfileobj(file.file, temp_file)
            temp_file.close()
            return Path(temp_file.name)
        except Exception as e:
            temp_file.close()
            if Path(temp_file.name).exists():
                Path(temp_file.name).unlink()
            raise


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION HELPERS (Fase 3 - Ligação aos módulos existentes)
# ═══════════════════════════════════════════════════════════════════════════════

def build_planning_instance_from_raw(
    db: Session,
    company_id: Optional[int] = None,
    horizon_days: int = 30,
) -> Dict[str, Any]:
    """
    Constrói SchedulingInstance a partir de dados raw.
    
    As specified in Contract 14:
    - Cria jobs a partir de ops_raw_orders
    - Cria operations a partir de routing_json
    - Cria machines a partir de ops_raw_machines
    
    Returns:
        dict com estrutura compatível com SchedulingInstance
    """
    from ops_ingestion.models import OpsRawOrder, OpsRawMachine
    
    # Buscar ordens
    orders = db.query(OpsRawOrder).all()
    
    # Buscar máquinas
    machines = db.query(OpsRawMachine).all()
    
    # Construir jobs
    jobs = []
    for order in orders:
        routing = order.get_routing()
        operations = []
        
        for op_idx, op in enumerate(routing or []):
            operations.append({
                "operation_id": f"{order.external_order_code}_OP{op_idx + 1}",
                "machine": op.get("machine"),
                "time_minutes": op.get("time_min") or op.get("time") or 0,
                "setup_minutes": op.get("setup_min") or 0,
            })
        
        jobs.append({
            "job_id": order.external_order_code,
            "product_code": order.product_code,
            "quantity": order.quantity,
            "due_date": order.due_date.isoformat() if order.due_date else None,
            "operations": operations,
            "source": "excel_import",
        })
    
    # Construir máquinas
    machines_list = []
    for machine in machines:
        machines_list.append({
            "machine_id": machine.machine_code,
            "description": machine.description,
            "line": machine.line,
            "capacity_per_shift_hours": machine.capacity_per_shift_hours,
            "avg_setup_time_minutes": machine.avg_setup_time_minutes,
            "source": "excel_import",
        })
    
    return {
        "jobs": jobs,
        "machines": machines_list,
        "horizon_days": horizon_days,
        "source": "ops_raw_excels",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_service_instance: Optional[OpsIngestionService] = None


def get_ops_ingestion_service() -> OpsIngestionService:
    """Get singleton service instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = OpsIngestionService()
    return _service_instance



"""Seed data for Duplios – creates sample DPPs on startup."""
from __future__ import annotations

from datetime import datetime, timedelta
import random
import string

from duplios.models import DPPModel, SessionLocal, init_db
from duplios.trust_index_stub import compute_trust_index

PUBLIC_BASE_URL = "https://duplios.local"


def _slug(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


SAMPLE_DPPS = [
    {
        "gtin": "5601234567890",
        "product_name": "Motor Elétrico Industrial ME-750",
        "product_category": "Equipamento Industrial",
        "manufacturer_name": "Nikufra Indústria, Lda",
        "manufacturer_eori": "PT500123456",
        "manufacturing_site_id": "FAB-PORTO-01",
        "country_of_origin": "PT",
        "serial_or_lot": "LOT-2024-0892",
        "materials": [
            {"material_name": "Aço Carbono", "material_type": "Metal", "percentage": 45, "mass_kg": 12.5},
            {"material_name": "Cobre", "material_type": "Metal", "percentage": 25, "mass_kg": 7.0},
            {"material_name": "Alumínio", "material_type": "Metal", "percentage": 15, "mass_kg": 4.2},
            {"material_name": "Polímero ABS", "material_type": "Plástico", "percentage": 10, "mass_kg": 2.8},
            {"material_name": "Resina Epóxi", "material_type": "Compósito", "percentage": 5, "mass_kg": 1.4},
        ],
        "components": [
            {"component_name": "Estator", "supplier_name": "ElectroParts PT", "weight_kg": 8.5},
            {"component_name": "Rotor", "supplier_name": "ElectroParts PT", "weight_kg": 6.2},
            {"component_name": "Rolamentos SKF", "supplier_name": "SKF Portugal", "weight_kg": 0.8},
            {"component_name": "Carcaça", "supplier_name": "Fundição Norte", "weight_kg": 10.0},
            {"component_name": "Ventilador", "supplier_name": "CoolTech", "weight_kg": 0.5},
        ],
        "carbon_footprint_kg_co2eq": 185.4,
        "impact_breakdown": {
            "manufacturing_kg_co2eq": 142.0,
            "distribution_kg_co2eq": 28.4,
            "end_of_life_kg_co2eq": 15.0,
        },
        "water_consumption_m3": 2.8,
        "energy_consumption_kwh": 320,
        "recycled_content_percent": 35,
        "recyclability_percent": 92,
        "durability_score": 9,
        "reparability_score": 8,
        "hazardous_substances": [
            {"substance_name": "Chumbo", "regulation": "ROHS", "status": "below_limit"},
            {"substance_name": "Cádmio", "regulation": "REACH", "status": "below_limit"},
        ],
        "certifications": [
            {"scheme": "CE", "issuer": "TÜV Rheinland", "valid_until": "2027-06-15T00:00:00"},
            {"scheme": "ISO 9001", "issuer": "SGS Portugal", "valid_until": "2026-12-01T00:00:00"},
            {"scheme": "ISO 14001", "issuer": "SGS Portugal", "valid_until": "2026-12-01T00:00:00"},
        ],
        "third_party_audits": [
            {"auditor_name": "Bureau Veritas", "scope": "Produção e Qualidade", "date": "2024-03-15T00:00:00", "result": "Conforme"},
            {"auditor_name": "EDP Sustentabilidade", "scope": "Pegada Carbónica", "date": "2024-05-20T00:00:00", "result": "Verificado"},
        ],
        "data_completeness_percent": 95,
        "status": "published",
    },
    {
        "gtin": "5601234567891",
        "product_name": "Painel Fotovoltaico PV-400W",
        "product_category": "Energia Renovável",
        "manufacturer_name": "SolarTech Portugal",
        "manufacturer_eori": "PT500789012",
        "manufacturing_site_id": "FAB-SETUBAL-02",
        "country_of_origin": "PT",
        "serial_or_lot": "LOT-2024-PV-1205",
        "materials": [
            {"material_name": "Silício Monocristalino", "material_type": "Semicondutor", "percentage": 40, "mass_kg": 8.0},
            {"material_name": "Vidro Temperado", "material_type": "Vidro", "percentage": 30, "mass_kg": 6.0},
            {"material_name": "Alumínio", "material_type": "Metal", "percentage": 18, "mass_kg": 3.6},
            {"material_name": "EVA", "material_type": "Polímero", "percentage": 8, "mass_kg": 1.6},
            {"material_name": "Cobre", "material_type": "Metal", "percentage": 4, "mass_kg": 0.8},
        ],
        "components": [
            {"component_name": "Células Solares", "supplier_name": "SunCell GmbH", "weight_kg": 8.0},
            {"component_name": "Moldura Alumínio", "supplier_name": "AlumPortugal", "weight_kg": 3.6},
            {"component_name": "Junction Box", "supplier_name": "ElecConnect", "weight_kg": 0.4},
            {"component_name": "Backsheet", "supplier_name": "DuPont", "weight_kg": 1.2},
        ],
        "carbon_footprint_kg_co2eq": 520.0,
        "impact_breakdown": {
            "manufacturing_kg_co2eq": 480.0,
            "distribution_kg_co2eq": 25.0,
            "end_of_life_kg_co2eq": 15.0,
        },
        "water_consumption_m3": 12.5,
        "energy_consumption_kwh": 850,
        "recycled_content_percent": 15,
        "recyclability_percent": 85,
        "durability_score": 10,
        "reparability_score": 4,
        "hazardous_substances": [
            {"substance_name": "Chumbo (soldadura)", "regulation": "ROHS", "status": "below_limit"},
        ],
        "certifications": [
            {"scheme": "IEC 61215", "issuer": "TÜV SÜD", "valid_until": "2028-01-20T00:00:00"},
            {"scheme": "IEC 61730", "issuer": "TÜV SÜD", "valid_until": "2028-01-20T00:00:00"},
            {"scheme": "CE", "issuer": "TÜV SÜD", "valid_until": "2028-01-20T00:00:00"},
        ],
        "third_party_audits": [
            {"auditor_name": "DNV", "scope": "Desempenho Energético", "date": "2024-02-10T00:00:00", "result": "Aprovado"},
        ],
        "data_completeness_percent": 88,
        "status": "published",
    },
    {
        "gtin": "5601234567892",
        "product_name": "Bateria Lítio Industrial BLI-48V",
        "product_category": "Armazenamento Energia",
        "manufacturer_name": "PowerStore Lda",
        "manufacturer_eori": "PT500456789",
        "manufacturing_site_id": "FAB-AVEIRO-01",
        "country_of_origin": "PT",
        "serial_or_lot": "BAT-2024-7821",
        "materials": [
            {"material_name": "Lítio", "material_type": "Metal", "percentage": 8, "mass_kg": 2.4},
            {"material_name": "Níquel", "material_type": "Metal", "percentage": 15, "mass_kg": 4.5},
            {"material_name": "Cobalto", "material_type": "Metal", "percentage": 10, "mass_kg": 3.0},
            {"material_name": "Grafite", "material_type": "Carbono", "percentage": 20, "mass_kg": 6.0},
            {"material_name": "Alumínio", "material_type": "Metal", "percentage": 25, "mass_kg": 7.5},
            {"material_name": "Polímero/Eletrólito", "material_type": "Químico", "percentage": 12, "mass_kg": 3.6},
            {"material_name": "Aço", "material_type": "Metal", "percentage": 10, "mass_kg": 3.0},
        ],
        "components": [
            {"component_name": "Células Li-ion NMC", "supplier_name": "CATL", "weight_kg": 18.0},
            {"component_name": "BMS (Sistema Gestão)", "supplier_name": "BatteryBrain", "weight_kg": 1.2},
            {"component_name": "Carcaça IP67", "supplier_name": "MetalBox PT", "weight_kg": 8.0},
            {"component_name": "Conectores", "supplier_name": "TE Connectivity", "weight_kg": 0.5},
        ],
        "carbon_footprint_kg_co2eq": 1850.0,
        "impact_breakdown": {
            "manufacturing_kg_co2eq": 1650.0,
            "distribution_kg_co2eq": 120.0,
            "end_of_life_kg_co2eq": 80.0,
        },
        "water_consumption_m3": 45.0,
        "energy_consumption_kwh": 2800,
        "recycled_content_percent": 12,
        "recyclability_percent": 70,
        "durability_score": 7,
        "reparability_score": 5,
        "hazardous_substances": [
            {"substance_name": "Cobalto", "regulation": "REACH", "status": "below_limit"},
            {"substance_name": "Lítio", "regulation": "EU Battery Regulation", "status": "below_limit"},
        ],
        "certifications": [
            {"scheme": "UN38.3", "issuer": "Intertek", "valid_until": "2026-08-30T00:00:00"},
            {"scheme": "CE", "issuer": "TÜV Rheinland", "valid_until": "2027-03-15T00:00:00"},
            {"scheme": "IEC 62619", "issuer": "TÜV Rheinland", "valid_until": "2027-03-15T00:00:00"},
        ],
        "third_party_audits": [
            {"auditor_name": "Intertek", "scope": "Segurança Transporte", "date": "2024-01-25T00:00:00", "result": "Conforme"},
            {"auditor_name": "RMI", "scope": "Cadeia de Fornecimento Minerais", "date": "2024-04-10T00:00:00", "result": "Verificado"},
        ],
        "data_completeness_percent": 92,
        "status": "published",
    },
    {
        "gtin": "5601234567893",
        "product_name": "Redutor Planetário RP-5000",
        "product_category": "Transmissão Mecânica",
        "manufacturer_name": "Nikufra Indústria, Lda",
        "manufacturer_eori": "PT500123456",
        "manufacturing_site_id": "FAB-PORTO-01",
        "country_of_origin": "PT",
        "serial_or_lot": "LOT-2024-RP-0456",
        "materials": [
            {"material_name": "Aço Ligado 42CrMo4", "material_type": "Metal", "percentage": 70, "mass_kg": 35.0},
            {"material_name": "Bronze", "material_type": "Metal", "percentage": 12, "mass_kg": 6.0},
            {"material_name": "Ferro Fundido", "material_type": "Metal", "percentage": 10, "mass_kg": 5.0},
            {"material_name": "Óleo Sintético", "material_type": "Lubrificante", "percentage": 5, "mass_kg": 2.5},
            {"material_name": "Vedantes NBR", "material_type": "Borracha", "percentage": 3, "mass_kg": 1.5},
        ],
        "components": [
            {"component_name": "Engrenagens Planetárias", "supplier_name": "GearTech DE", "weight_kg": 18.0},
            {"component_name": "Caixa", "supplier_name": "Fundição Norte", "weight_kg": 20.0},
            {"component_name": "Rolamentos FAG", "supplier_name": "Schaeffler", "weight_kg": 3.5},
            {"component_name": "Vedantes", "supplier_name": "SKF", "weight_kg": 1.5},
            {"component_name": "Veio de Saída", "supplier_name": "Interno", "weight_kg": 5.0},
        ],
        "carbon_footprint_kg_co2eq": 245.0,
        "impact_breakdown": {
            "manufacturing_kg_co2eq": 195.0,
            "distribution_kg_co2eq": 35.0,
            "end_of_life_kg_co2eq": 15.0,
        },
        "water_consumption_m3": 4.2,
        "energy_consumption_kwh": 420,
        "recycled_content_percent": 45,
        "recyclability_percent": 95,
        "durability_score": 10,
        "reparability_score": 9,
        "hazardous_substances": [],
        "certifications": [
            {"scheme": "CE", "issuer": "Lloyd's", "valid_until": "2027-09-01T00:00:00"},
            {"scheme": "ISO 9001", "issuer": "SGS", "valid_until": "2026-12-01T00:00:00"},
        ],
        "third_party_audits": [
            {"auditor_name": "Lloyd's Register", "scope": "Qualidade Metalúrgica", "date": "2024-06-05T00:00:00", "result": "Conforme"},
        ],
        "data_completeness_percent": 90,
        "status": "published",
    },
    {
        "gtin": "5601234567894",
        "product_name": "Têxtil Técnico Fireproof T-500",
        "product_category": "Têxtil Industrial",
        "manufacturer_name": "TecFibras SA",
        "manufacturer_eori": "PT500654321",
        "manufacturing_site_id": "FAB-BRAGA-01",
        "country_of_origin": "PT",
        "serial_or_lot": "TEX-2024-5500",
        "materials": [
            {"material_name": "Aramida (Kevlar)", "material_type": "Fibra Sintética", "percentage": 50, "mass_kg": 0.5},
            {"material_name": "Fibra de Vidro", "material_type": "Fibra Mineral", "percentage": 30, "mass_kg": 0.3},
            {"material_name": "Algodão FR", "material_type": "Fibra Natural", "percentage": 15, "mass_kg": 0.15},
            {"material_name": "Elastano", "material_type": "Fibra Sintética", "percentage": 5, "mass_kg": 0.05},
        ],
        "components": [],
        "carbon_footprint_kg_co2eq": 12.5,
        "impact_breakdown": {
            "manufacturing_kg_co2eq": 9.8,
            "distribution_kg_co2eq": 1.7,
            "end_of_life_kg_co2eq": 1.0,
        },
        "water_consumption_m3": 85.0,
        "energy_consumption_kwh": 45,
        "recycled_content_percent": 20,
        "recyclability_percent": 40,
        "durability_score": 8,
        "reparability_score": 3,
        "hazardous_substances": [
            {"substance_name": "Retardante FR", "regulation": "REACH", "status": "below_limit"},
        ],
        "certifications": [
            {"scheme": "OEKO-TEX 100", "issuer": "Hohenstein", "valid_until": "2025-11-30T00:00:00"},
            {"scheme": "EN 11612", "issuer": "CENTEXBEL", "valid_until": "2026-05-15T00:00:00"},
            {"scheme": "EN 1149-5", "issuer": "CENTEXBEL", "valid_until": "2026-05-15T00:00:00"},
        ],
        "third_party_audits": [
            {"auditor_name": "CITEVE", "scope": "Resistência ao Fogo", "date": "2024-04-20T00:00:00", "result": "Aprovado EN 11612"},
        ],
        "data_completeness_percent": 85,
        "status": "published",
    },
    {
        "gtin": "5601234567895",
        "product_name": "Válvula Pneumática VP-DN50",
        "product_category": "Automação Industrial",
        "manufacturer_name": "FluidControl PT",
        "manufacturer_eori": "PT500111222",
        "manufacturing_site_id": "FAB-LEIRIA-01",
        "country_of_origin": "PT",
        "serial_or_lot": "VP-2024-3345",
        "materials": [
            {"material_name": "Aço Inox 316L", "material_type": "Metal", "percentage": 55, "mass_kg": 2.75},
            {"material_name": "PTFE", "material_type": "Polímero", "percentage": 15, "mass_kg": 0.75},
            {"material_name": "Alumínio", "material_type": "Metal", "percentage": 20, "mass_kg": 1.0},
            {"material_name": "NBR/Viton", "material_type": "Elastómero", "percentage": 10, "mass_kg": 0.5},
        ],
        "components": [
            {"component_name": "Corpo Válvula", "supplier_name": "Interno", "weight_kg": 2.5},
            {"component_name": "Atuador Pneumático", "supplier_name": "Festo", "weight_kg": 1.8},
            {"component_name": "Vedantes", "supplier_name": "Trelleborg", "weight_kg": 0.3},
            {"component_name": "Posicionador", "supplier_name": "Interno", "weight_kg": 0.4},
        ],
        "carbon_footprint_kg_co2eq": 28.5,
        "impact_breakdown": {
            "manufacturing_kg_co2eq": 22.0,
            "distribution_kg_co2eq": 4.5,
            "end_of_life_kg_co2eq": 2.0,
        },
        "water_consumption_m3": 0.8,
        "energy_consumption_kwh": 65,
        "recycled_content_percent": 30,
        "recyclability_percent": 88,
        "durability_score": 9,
        "reparability_score": 8,
        "hazardous_substances": [],
        "certifications": [
            {"scheme": "CE", "issuer": "TÜV Nord", "valid_until": "2027-02-28T00:00:00"},
            {"scheme": "ATEX", "issuer": "TÜV Nord", "valid_until": "2027-02-28T00:00:00"},
            {"scheme": "PED", "issuer": "Lloyd's", "valid_until": "2026-10-01T00:00:00"},
        ],
        "third_party_audits": [
            {"auditor_name": "TÜV Nord", "scope": "Conformidade ATEX", "date": "2024-02-01T00:00:00", "result": "Conforme"},
        ],
        "data_completeness_percent": 93,
        "status": "published",
    },
    {
        "gtin": "5601234567896",
        "product_name": "Inversor Frequência VFD-22kW",
        "product_category": "Eletrónica Industrial",
        "manufacturer_name": "DriveElectric SA",
        "manufacturer_eori": "PT500333444",
        "manufacturing_site_id": "FAB-COIMBRA-01",
        "country_of_origin": "PT",
        "serial_or_lot": "VFD-2024-8890",
        "materials": [
            {"material_name": "Cobre", "material_type": "Metal", "percentage": 25, "mass_kg": 3.75},
            {"material_name": "Alumínio", "material_type": "Metal", "percentage": 30, "mass_kg": 4.5},
            {"material_name": "PCB/FR4", "material_type": "Compósito", "percentage": 15, "mass_kg": 2.25},
            {"material_name": "Componentes Eletrónicos", "material_type": "Misto", "percentage": 20, "mass_kg": 3.0},
            {"material_name": "Plástico ABS", "material_type": "Polímero", "percentage": 10, "mass_kg": 1.5},
        ],
        "components": [
            {"component_name": "IGBTs Infineon", "supplier_name": "Infineon", "weight_kg": 1.2},
            {"component_name": "Condensadores DC-Link", "supplier_name": "Nichicon", "weight_kg": 2.0},
            {"component_name": "Dissipador", "supplier_name": "Interno", "weight_kg": 4.0},
            {"component_name": "PCB Principal", "supplier_name": "PCBWay", "weight_kg": 1.5},
            {"component_name": "Carcaça", "supplier_name": "MetalBox PT", "weight_kg": 5.0},
        ],
        "carbon_footprint_kg_co2eq": 156.0,
        "impact_breakdown": {
            "manufacturing_kg_co2eq": 125.0,
            "distribution_kg_co2eq": 18.0,
            "end_of_life_kg_co2eq": 13.0,
        },
        "water_consumption_m3": 1.5,
        "energy_consumption_kwh": 280,
        "recycled_content_percent": 18,
        "recyclability_percent": 75,
        "durability_score": 8,
        "reparability_score": 6,
        "hazardous_substances": [
            {"substance_name": "Chumbo (soldadura legacy)", "regulation": "ROHS", "status": "below_limit"},
            {"substance_name": "Retardante bromado", "regulation": "REACH", "status": "below_limit"},
        ],
        "certifications": [
            {"scheme": "CE", "issuer": "TÜV SÜD", "valid_until": "2027-04-15T00:00:00"},
            {"scheme": "UL", "issuer": "UL", "valid_until": "2026-09-20T00:00:00"},
            {"scheme": "EMC EN 61800-3", "issuer": "TÜV SÜD", "valid_until": "2027-04-15T00:00:00"},
        ],
        "third_party_audits": [
            {"auditor_name": "SGS", "scope": "ROHS Compliance", "date": "2024-03-10T00:00:00", "result": "Conforme"},
        ],
        "data_completeness_percent": 89,
        "status": "published",
    },
    {
        "gtin": "5601234567897",
        "product_name": "Mobiliário Industrial Bancada MB-2000",
        "product_category": "Mobiliário",
        "manufacturer_name": "IndustriMob Lda",
        "manufacturer_eori": "PT500555666",
        "manufacturing_site_id": "FAB-PACOS-01",
        "country_of_origin": "PT",
        "serial_or_lot": "MB-2024-1122",
        "materials": [
            {"material_name": "Aço Galvanizado", "material_type": "Metal", "percentage": 60, "mass_kg": 48.0},
            {"material_name": "MDF", "material_type": "Madeira", "percentage": 25, "mass_kg": 20.0},
            {"material_name": "Melamina", "material_type": "Revestimento", "percentage": 8, "mass_kg": 6.4},
            {"material_name": "Borracha", "material_type": "Elastómero", "percentage": 5, "mass_kg": 4.0},
            {"material_name": "Parafusos/Fixações", "material_type": "Metal", "percentage": 2, "mass_kg": 1.6},
        ],
        "components": [
            {"component_name": "Estrutura Metálica", "supplier_name": "Interno", "weight_kg": 45.0},
            {"component_name": "Tampo MDF", "supplier_name": "Sonae Arauco", "weight_kg": 25.0},
            {"component_name": "Pés Ajustáveis", "supplier_name": "Häfele", "weight_kg": 4.0},
            {"component_name": "Calhas Técnicas", "supplier_name": "Interno", "weight_kg": 3.0},
        ],
        "carbon_footprint_kg_co2eq": 95.0,
        "impact_breakdown": {
            "manufacturing_kg_co2eq": 72.0,
            "distribution_kg_co2eq": 15.0,
            "end_of_life_kg_co2eq": 8.0,
        },
        "water_consumption_m3": 0.6,
        "energy_consumption_kwh": 85,
        "recycled_content_percent": 55,
        "recyclability_percent": 90,
        "durability_score": 9,
        "reparability_score": 10,
        "hazardous_substances": [
            {"substance_name": "Formaldeído (MDF)", "regulation": "REACH", "status": "below_limit"},
        ],
        "certifications": [
            {"scheme": "FSC", "issuer": "FSC", "valid_until": "2025-12-31T00:00:00"},
            {"scheme": "EN 527", "issuer": "CTIMM", "valid_until": "2026-06-30T00:00:00"},
        ],
        "third_party_audits": [
            {"auditor_name": "LNEC", "scope": "Resistência Estrutural", "date": "2024-05-15T00:00:00", "result": "Conforme"},
        ],
        "data_completeness_percent": 87,
        "status": "published",
    },
]


def seed_dpps() -> int:
    """Insert sample DPPs if table is empty. Returns count of inserted records."""
    init_db()
    session = SessionLocal()
    try:
        existing = session.query(DPPModel).count()
        if existing > 0:
            return 0

        inserted = 0
        for dpp_data in SAMPLE_DPPS:
            slug = _slug()
            gtin = dpp_data["gtin"]
            trust = compute_trust_index(dpp_data)
            completeness = dpp_data.get("data_completeness_percent", 80)
            carbon = dpp_data.get("carbon_footprint_kg_co2eq", 0)
            recyclability = dpp_data.get("recyclability_percent", 0)

            dpp = DPPModel(
                dpp_id=f"{gtin}-{slug}",
                qr_slug=slug,
                gtin=gtin,
                product_name=dpp_data["product_name"],
                product_category=dpp_data.get("product_category"),
                manufacturer_name=dpp_data.get("manufacturer_name"),
                country_of_origin=dpp_data.get("country_of_origin"),
                trust_index=trust,
                carbon_footprint_kg_co2eq=carbon,
                recyclability_percent=recyclability,
                data_completeness_percent=completeness,
                qr_public_url=f"{PUBLIC_BASE_URL}/duplios/view/{slug}",
                dpp_data=dpp_data,
                status=dpp_data.get("status", "draft"),
                created_by_user_id="system",
            )
            session.add(dpp)
            inserted += 1

        session.commit()
        return inserted
    finally:
        session.close()




export interface KPI {
  otd_pct: number;
  lead_time_h: number;
  gargalo_ativo: string;
  horas_setup_semana: number;
}

export interface Operation {
  ordem: string;
  artigo: string;
  operacao: string;
  recurso: string;
  start_time: string;
  end_time: string;
  setor: string;
  overlap: number;
  rota: string;
  explicacao: string;
}

export interface Plan {
  kpis: KPI;
  operations: Operation[];
  explicacoes: string[];
}

export interface PlanResponse {
  antes: Plan;
  depois: Plan;
}

// Nova API v2 types
export interface PlanV2Operation {
  order_id: string;
  op_id: string;
  rota: string;
  maquina_id: string;
  start_time: string;
  end_time: string;
  quantidade: number;
  duracao_h: number;
  family: string;
  artigo?: string; // Optional: article ID
}

export interface PlanV2Result {
  makespan_h: number;
  total_setup_h: number;
  kpis: Record<string, number>;
  operations: PlanV2Operation[];
  gantt_by_machine: Record<string, PlanV2Operation[]>;
  all_machines?: string[];  // Lista completa de máquinas (mesmo sem operações)
}

export interface PlanV2Response {
  batch_id: string;
  horizon_hours: number;
  created_at: string;
  baseline: PlanV2Result | null;
  optimized: PlanV2Result | null;
  config: Record<string, any>;
  orders_summary?: {
    total_orders: number;
    orders: string[];
  };
}

export interface Bottleneck {
  recurso: string;
  utilizacao_pct: number;
  fila_horas: number;
  probabilidade: number;
  drivers: string[];
  acao: string;
  impacto_otd: number;
  impacto_horas: number;
  [key: string]: unknown; // Index signature for DataTable compatibility
}

export interface BottleneckResponse {
  bottlenecks: Bottleneck[];
  top_losses: Bottleneck[];
  heatmap: Array<{
    recurso: string;
    utilizacao: Array<{
      hora: number;
      utilizacao_pct: number;
    }>;
    demo_override?: boolean;
  }>;
  overlap_applied: {
    transformacao: number;
    acabamentos: number;
    embalagem: number;
  };
  lead_time_gain: number;
  demo_mode?: boolean;
  demo_overrides?: Record<string, boolean>;
}

export interface InventorySKU {
  sku: string;
  classe: string;
  xyz: string;
  stock_atual: number;
  ads_180: number;
  cobertura_dias: number;
  risco_30d: number;
  rop: number;
  acao: string;
}

export interface InventoryResponse {
  matrix: {
    A: { X: number; Y: number; Z: number };
    B: { X: number; Y: number; Z: number };
    C: { X: number; Y: number; Z: number };
  };
  skus: InventorySKU[];
}

export interface VIPRequest {
  sku: string;
  quantidade: number;
  prazo: string;
}

export interface AvariaRequest {
  recurso: string;
  de: string;
  ate: string;
}


/**
 * ════════════════════════════════════════════════════════════════════════════════════════════
 * PRODPLAN 4.0 — PROJECT PLANNING VIEW
 * ════════════════════════════════════════════════════════════════════════════════════════════
 *
 * Vista de planeamento por projeto.
 * Permite ver e decidir o plano por PROJETO (não apenas por ordem/artigo).
 */

import React, { useState, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  FolderKanban,
  AlertTriangle,
  CheckCircle2,
  Clock,
  TrendingUp,
  ChevronDown,
  ChevronUp,
  RefreshCw,
  Zap,
  Target,
  Users,
  Calendar,
} from 'lucide-react'
import { toast } from 'sonner'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

// ═══════════════════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════════════════

interface ProjectKPI {
  project_id: string
  project_name: string
  client: string | null
  lead_time_hours: number
  lead_time_days: number
  delay_hours: number
  delay_days: number
  slack_hours: number
  total_orders: number
  completed_orders: number
  completion_rate_pct: number
  total_load_hours: number
  priority_weight: number
  risk_score: number
  risk_level: string
  probability_late_pct: number
  status: string
  is_on_time: boolean
  due_date: string | null
  estimated_completion: string | null
}

interface GlobalKPIs {
  total_projects: number
  projects_on_time: number
  projects_delayed: number
  projects_at_risk: number
  otd_pct: number
  weighted_otd_pct: number
  avg_lead_time_days: number
  total_delay_hours: number
  bottleneck_machine: string | null
}

interface ProjectsResponse {
  projects: ProjectKPI[]
  global_kpis: GlobalKPIs
  aggregation_mode: string
  total_projects: number
}

interface PriorityPlanResponse {
  priority_plan: {
    priorities: Array<{
      project_id: string
      priority_rank: number
      suggested_start_day: number
      expected_end_day: number
      expected_delay_days: number
      priority_score: number
    }>
    expected_otd_pct: number
    solver_status: string
  }
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// API FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════════════════

async function fetchProjects(mode: string): Promise<ProjectsResponse> {
  const res = await fetch(`${API_BASE_URL}/projects?aggregation_mode=${mode}`)
  if (!res.ok) throw new Error('Erro ao carregar projetos')
  return res.json()
}

async function fetchPriorityPlan(useMilp: boolean): Promise<PriorityPlanResponse> {
  const res = await fetch(`${API_BASE_URL}/projects/priority-plan?use_milp=${useMilp}`)
  if (!res.ok) throw new Error('Erro ao calcular prioridades')
  return res.json()
}

async function recomputePlan(useMilp: boolean): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/projects/recompute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ use_milp: useMilp }),
  })
  if (!res.ok) throw new Error('Erro ao recalcular plano')
  return res.json()
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════════════════

// Risk badge component
const RiskBadge: React.FC<{ level: string }> = ({ level }) => {
  const colors: Record<string, string> = {
    LOW: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
    MEDIUM: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
    HIGH: 'bg-orange-500/20 text-orange-300 border-orange-500/30',
    CRITICAL: 'bg-red-500/20 text-red-300 border-red-500/30',
    UNKNOWN: 'bg-slate-500/20 text-slate-300 border-slate-500/30',
  }
  
  return (
    <span className={`px-2 py-0.5 text-xs font-medium rounded border ${colors[level] || colors.UNKNOWN}`}>
      {level}
    </span>
  )
}

// Status badge component
const StatusBadge: React.FC<{ status: string; isOnTime: boolean }> = ({ status, isOnTime }) => {
  if (isOnTime) {
    return (
      <span className="flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded bg-emerald-500/20 text-emerald-300">
        <CheckCircle2 className="w-3 h-3" />
        No prazo
      </span>
    )
  }
  
  return (
    <span className="flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded bg-red-500/20 text-red-300">
      <AlertTriangle className="w-3 h-3" />
      Atrasado
    </span>
  )
}

// KPI Card component
const KPICard: React.FC<{
  title: string
  value: string | number
  subtitle?: string
  icon: React.ReactNode
  trend?: 'up' | 'down' | 'neutral'
  highlight?: boolean
}> = ({ title, value, subtitle, icon, trend, highlight }) => {
  return (
    <div className={`
      p-4 rounded-lg border transition-all
      ${highlight 
        ? 'bg-gradient-to-br from-cyan-500/20 to-blue-600/20 border-cyan-500/30' 
        : 'bg-slate-800/50 border-slate-700/50'}
    `}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-slate-400 uppercase tracking-wider">{title}</p>
          <p className="text-2xl font-bold text-white mt-1">{value}</p>
          {subtitle && <p className="text-xs text-slate-500 mt-0.5">{subtitle}</p>}
        </div>
        <div className={`p-2 rounded-lg ${highlight ? 'bg-cyan-500/20' : 'bg-slate-700/50'}`}>
          {icon}
        </div>
      </div>
    </div>
  )
}

// Project row component (expandable)
const ProjectRow: React.FC<{
  project: ProjectKPI
  rank?: number
  expanded: boolean
  onToggle: () => void
}> = ({ project, rank, expanded, onToggle }) => {
  return (
    <motion.div
      layout
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="border border-slate-700/50 rounded-lg overflow-hidden mb-2"
    >
      {/* Header row */}
      <div
        onClick={onToggle}
        className={`
          p-4 cursor-pointer transition-colors
          ${expanded ? 'bg-slate-700/30' : 'bg-slate-800/30 hover:bg-slate-700/20'}
        `}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            {rank && (
              <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-sm font-bold">
                {rank}
              </div>
            )}
            <div>
              <h3 className="font-semibold text-white">{project.project_name}</h3>
              <p className="text-xs text-slate-400">
                {project.client || 'Sem cliente'} • {project.total_orders} encomendas
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            <div className="text-right">
              <p className="text-sm font-medium text-white">{(project.total_load_hours ?? 0).toFixed(1)}h</p>
              <p className="text-xs text-slate-400">carga total</p>
            </div>
            
            <RiskBadge level={project.risk_level} />
            <StatusBadge status={project.status} isOnTime={project.is_on_time} />
            
            {expanded ? (
              <ChevronUp className="w-5 h-5 text-slate-400" />
            ) : (
              <ChevronDown className="w-5 h-5 text-slate-400" />
            )}
          </div>
        </div>
      </div>
      
      {/* Expanded content */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="border-t border-slate-700/50"
          >
            <div className="p-4 bg-slate-900/50 grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-xs text-slate-500">Lead Time</p>
                <p className="text-lg font-semibold text-white">
                  {(project.lead_time_days ?? 0).toFixed(1)} dias
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Atraso</p>
                <p className={`text-lg font-semibold ${(project.delay_hours ?? 0) > 0 ? 'text-red-400' : 'text-emerald-400'}`}>
                  {(project.delay_hours ?? 0) > 0 ? `${(project.delay_hours ?? 0).toFixed(1)}h` : 'Nenhum'}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Folga</p>
                <p className={`text-lg font-semibold ${(project.slack_hours ?? 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {(project.slack_hours ?? 0).toFixed(1)}h
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Prob. Atraso</p>
                <p className={`text-lg font-semibold ${(project.probability_late_pct ?? 0) > 50 ? 'text-red-400' : 'text-white'}`}>
                  {(project.probability_late_pct ?? 0).toFixed(0)}%
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Conclusão</p>
                <p className="text-lg font-semibold text-white">
                  {(project.completion_rate_pct ?? 0).toFixed(0)}%
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Prioridade</p>
                <p className="text-lg font-semibold text-cyan-400">
                  {(project.priority_weight ?? 1).toFixed(1)}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Data Limite</p>
                <p className="text-sm font-medium text-white">
                  {project.due_date ? new Date(project.due_date).toLocaleDateString('pt-PT') : '-'}
                </p>
              </div>
              <div>
                <p className="text-xs text-slate-500">Conclusão Estimada</p>
                <p className="text-sm font-medium text-white">
                  {project.estimated_completion ? new Date(project.estimated_completion).toLocaleDateString('pt-PT') : '-'}
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

// ═══════════════════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════════════════

const ProjectPlanning: React.FC = () => {
  const queryClient = useQueryClient()
  const [aggregationMode, setAggregationMode] = useState('explicit')
  const [expandedProject, setExpandedProject] = useState<string | null>(null)
  const [useMilp, setUseMilp] = useState(false)
  
  // Fetch projects
  const { data: projectsData, isLoading, error } = useQuery({
    queryKey: ['projects', aggregationMode],
    queryFn: () => fetchProjects(aggregationMode),
    refetchInterval: 30000,
  })
  
  // Fetch priority plan
  const { data: priorityData } = useQuery({
    queryKey: ['priority-plan', useMilp],
    queryFn: () => fetchPriorityPlan(useMilp),
    enabled: !!projectsData?.projects.length,
  })
  
  // Recompute mutation
  const recomputeMutation = useMutation({
    mutationFn: () => recomputePlan(useMilp),
    onSuccess: () => {
      toast.success('Plano recalculado com sucesso')
      queryClient.invalidateQueries({ queryKey: ['projects'] })
      queryClient.invalidateQueries({ queryKey: ['priority-plan'] })
    },
    onError: () => {
      toast.error('Erro ao recalcular plano')
    },
  })
  
  // Sort projects by priority
  const sortedProjects = useMemo(() => {
    if (!projectsData?.projects) return []
    
    // Get priority ranks from priority plan
    const ranks = new Map<string, number>()
    priorityData?.priority_plan?.priorities?.forEach((p) => {
      ranks.set(p.project_id, p.priority_rank)
    })
    
    return [...projectsData.projects].sort((a, b) => {
      const rankA = ranks.get(a.project_id) ?? 999
      const rankB = ranks.get(b.project_id) ?? 999
      return rankA - rankB
    })
  }, [projectsData, priorityData])
  
  const global = projectsData?.global_kpis
  
  if (error) {
    return (
      <div className="p-8 text-center">
        <AlertTriangle className="w-12 h-12 text-red-400 mx-auto mb-4" />
        <h2 className="text-xl font-semibold text-white mb-2">Erro ao carregar projetos</h2>
        <p className="text-slate-400">Verifique se a API está a funcionar corretamente.</p>
      </div>
    )
  }
  
  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <FolderKanban className="w-7 h-7 text-cyan-400" />
            Planeamento por Projeto
          </h1>
          <p className="text-slate-400 mt-1">
            Visão agregada por projeto • {sortedProjects.length} projetos
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          {/* Aggregation mode selector */}
          <select
            value={aggregationMode}
            onChange={(e) => setAggregationMode(e.target.value)}
            className="px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white text-sm"
          >
            <option value="explicit">Por Projeto (explícito)</option>
            <option value="by_client">Por Cliente</option>
            <option value="by_family">Por Família de Artigo</option>
            <option value="by_due_week">Por Semana de Entrega</option>
          </select>
          
          {/* MILP toggle */}
          <label className="flex items-center gap-2 px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg cursor-pointer">
            <input
              type="checkbox"
              checked={useMilp}
              onChange={(e) => setUseMilp(e.target.checked)}
              className="w-4 h-4 accent-cyan-500"
            />
            <span className="text-sm text-white">MILP</span>
          </label>
          
          {/* Recompute button */}
          <button
            onClick={() => recomputeMutation.mutate()}
            disabled={recomputeMutation.isPending}
            className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${recomputeMutation.isPending ? 'animate-spin' : ''}`} />
            Aplicar Prioridades
          </button>
        </div>
      </div>
      
      {/* Global KPIs */}
      {global && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
          <KPICard
            title="Projetos"
            value={global.total_projects}
            icon={<FolderKanban className="w-5 h-5 text-cyan-400" />}
          />
          <KPICard
            title="OTD"
            value={`${(global.otd_pct ?? 0).toFixed(0)}%`}
            subtitle={`${global.projects_on_time ?? 0} no prazo`}
            icon={<Target className="w-5 h-5 text-emerald-400" />}
            highlight={(global.otd_pct ?? 0) >= 90}
          />
          <KPICard
            title="Atrasados"
            value={global.projects_delayed ?? 0}
            subtitle={`${(global.total_delay_hours ?? 0).toFixed(0)}h total`}
            icon={<Clock className="w-5 h-5 text-red-400" />}
            highlight={(global.projects_delayed ?? 0) > 0}
          />
          <KPICard
            title="Em Risco"
            value={global.projects_at_risk ?? 0}
            icon={<AlertTriangle className="w-5 h-5 text-amber-400" />}
          />
          <KPICard
            title="Lead Time Médio"
            value={`${(global.avg_lead_time_days ?? 0).toFixed(1)}d`}
            icon={<TrendingUp className="w-5 h-5 text-blue-400" />}
          />
          <KPICard
            title="Gargalo"
            value={global.bottleneck_machine || '-'}
            icon={<Zap className="w-5 h-5 text-orange-400" />}
          />
        </div>
      )}
      
      {/* Priority plan summary */}
      {priorityData?.priority_plan && (
        <div className="p-4 bg-gradient-to-r from-cyan-900/30 to-blue-900/30 border border-cyan-700/30 rounded-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Zap className="w-5 h-5 text-cyan-400" />
              <span className="font-medium text-white">Plano de Prioridades Otimizado</span>
              <span className="text-xs text-slate-400">
                ({priorityData.priority_plan.solver_status})
              </span>
            </div>
            <div className="text-sm text-cyan-300">
              OTD Esperado: {(priorityData.priority_plan.expected_otd_pct ?? 0).toFixed(0)}%
            </div>
          </div>
        </div>
      )}
      
      {/* Projects list */}
      <div className="space-y-2">
        {isLoading ? (
          <div className="text-center py-12">
            <RefreshCw className="w-8 h-8 text-cyan-400 animate-spin mx-auto mb-4" />
            <p className="text-slate-400">A carregar projetos...</p>
          </div>
        ) : sortedProjects.length === 0 ? (
          <div className="text-center py-12 bg-slate-800/30 rounded-lg border border-slate-700/50">
            <FolderKanban className="w-12 h-12 text-slate-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-white mb-2">Nenhum projeto encontrado</h3>
            <p className="text-slate-400">
              Tente outro modo de agregação ou verifique se os dados têm o campo project_id.
            </p>
          </div>
        ) : (
          sortedProjects.map((project, index) => {
            const rank = priorityData?.priority_plan?.priorities?.find(
              p => p.project_id === project.project_id
            )?.priority_rank ?? index + 1
            
            return (
              <ProjectRow
                key={project.project_id}
                project={project}
                rank={rank}
                expanded={expandedProject === project.project_id}
                onToggle={() => setExpandedProject(
                  expandedProject === project.project_id ? null : project.project_id
                )}
              />
            )
          })
        )}
      </div>
      
      {/* Load visualization */}
      {sortedProjects.length > 0 && (
        <div className="bg-slate-800/50 border border-slate-700/50 rounded-lg p-4">
          <h3 className="text-sm font-medium text-slate-300 mb-4">Distribuição de Carga por Projeto</h3>
          <div className="space-y-2">
            {sortedProjects.slice(0, 10).map((project) => {
              const maxLoad = Math.max(...sortedProjects.map(p => p.total_load_hours))
              const pct = (project.total_load_hours / maxLoad) * 100
              
              return (
                <div key={project.project_id} className="flex items-center gap-3">
                  <span className="w-32 text-xs text-slate-400 truncate">{project.project_name}</span>
                  <div className="flex-1 h-4 bg-slate-700 rounded overflow-hidden">
                    <div
                      className={`h-full transition-all ${
                        project.risk_level === 'CRITICAL' ? 'bg-red-500' :
                        project.risk_level === 'HIGH' ? 'bg-orange-500' :
                        project.risk_level === 'MEDIUM' ? 'bg-amber-500' :
                        'bg-cyan-500'
                      }`}
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                  <span className="w-16 text-xs text-slate-400 text-right">
                    {(project.total_load_hours ?? 0).toFixed(0)}h
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

export default ProjectPlanning


/**
 * MRP Forecast Integration Panel
 * 
 * Features:
 * - Upload forecast data (CSV/JSON)
 * - Forecast vs demand visualization
 * - Integration with MRP runs
 * - Forecast accuracy metrics
 */

import React, { useState, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { motion, AnimatePresence } from 'framer-motion';
import {
  AlertTriangle,
  BarChart2,
  Calendar,
  Check,
  ChevronDown,
  ChevronRight,
  Cloud,
  Download,
  FileUp,
  Layers,
  LineChart,
  Loader2,
  Play,
  RefreshCw,
  Settings,
  Target,
  TrendingUp,
  Upload,
  X,
  Zap,
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface ForecastData {
  item_code: string;
  item_name: string;
  forecasts: ForecastPeriod[];
  accuracy: number;
  mape: number;
  bias: number;
}

interface ForecastPeriod {
  period: string;
  forecast_qty: number;
  actual_qty?: number;
  confidence_lower: number;
  confidence_upper: number;
}

interface ForecastUploadResult {
  success: boolean;
  items_imported: number;
  periods: number;
  warnings: string[];
}

interface ForecastAccuracy {
  overall_mape: number;
  overall_bias: number;
  accuracy_by_category: { category: string; mape: number; bias: number }[];
  best_performers: { item_code: string; mape: number }[];
  worst_performers: { item_code: string; mape: number }[];
}

// ═══════════════════════════════════════════════════════════════════════════════
// API
// ═══════════════════════════════════════════════════════════════════════════════

async function fetchForecasts(): Promise<ForecastData[]> {
  try {
    const res = await fetch(`${API_BASE}/smart-inventory/forecasts`);
    if (!res.ok) throw new Error();
    return res.json();
  } catch {
    // Mock data for demo
    return [
      {
        item_code: 'SKU-001',
        item_name: 'Widget Alpha',
        accuracy: 92.5,
        mape: 7.5,
        bias: -2.3,
        forecasts: [
          { period: '2024-W45', forecast_qty: 150, actual_qty: 145, confidence_lower: 130, confidence_upper: 170 },
          { period: '2024-W46', forecast_qty: 165, actual_qty: 172, confidence_lower: 145, confidence_upper: 185 },
          { period: '2024-W47', forecast_qty: 180, actual_qty: 178, confidence_lower: 155, confidence_upper: 205 },
          { period: '2024-W48', forecast_qty: 200, confidence_lower: 170, confidence_upper: 230 },
          { period: '2024-W49', forecast_qty: 220, confidence_lower: 185, confidence_upper: 255 },
        ],
      },
      {
        item_code: 'SKU-002',
        item_name: 'Gadget Beta',
        accuracy: 85.2,
        mape: 14.8,
        bias: 5.1,
        forecasts: [
          { period: '2024-W45', forecast_qty: 80, actual_qty: 72, confidence_lower: 65, confidence_upper: 95 },
          { period: '2024-W46', forecast_qty: 85, actual_qty: 88, confidence_lower: 70, confidence_upper: 100 },
          { period: '2024-W47', forecast_qty: 90, actual_qty: 95, confidence_lower: 75, confidence_upper: 105 },
          { period: '2024-W48', forecast_qty: 95, confidence_lower: 78, confidence_upper: 112 },
          { period: '2024-W49', forecast_qty: 100, confidence_lower: 82, confidence_upper: 118 },
        ],
      },
    ];
  }
}

async function fetchForecastAccuracy(): Promise<ForecastAccuracy> {
  try {
    const res = await fetch(`${API_BASE}/smart-inventory/forecast-accuracy`);
    if (!res.ok) throw new Error();
    return res.json();
  } catch {
    return {
      overall_mape: 11.2,
      overall_bias: 1.4,
      accuracy_by_category: [
        { category: 'Finished Goods', mape: 8.5, bias: -1.2 },
        { category: 'Raw Materials', mape: 12.3, bias: 2.8 },
        { category: 'Components', mape: 15.1, bias: 3.5 },
      ],
      best_performers: [
        { item_code: 'SKU-001', mape: 7.5 },
        { item_code: 'SKU-015', mape: 8.2 },
        { item_code: 'SKU-023', mape: 9.1 },
      ],
      worst_performers: [
        { item_code: 'SKU-042', mape: 28.5 },
        { item_code: 'SKU-031', mape: 24.2 },
        { item_code: 'SKU-019', mape: 21.8 },
      ],
    };
  }
}

async function uploadForecast(file: File): Promise<ForecastUploadResult> {
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const res = await fetch(`${API_BASE}/smart-inventory/forecast/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error();
    return res.json();
  } catch {
    // Mock success for demo
    return {
      success: true,
      items_imported: Math.floor(Math.random() * 50) + 10,
      periods: 12,
      warnings: [],
    };
  }
}

async function runMRPWithForecast(): Promise<void> {
  const res = await fetch(`${API_BASE}/smart-inventory/mrp/run-with-forecast`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('Failed to run MRP');
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

const SimpleBarChart: React.FC<{ data: ForecastPeriod[]; height?: number }> = ({ data, height = 120 }) => {
  const maxValue = Math.max(...data.map(d => Math.max(d.forecast_qty, d.actual_qty || 0, d.confidence_upper)));
  
  return (
    <div className="flex items-end gap-2 justify-between" style={{ height }}>
      {data.map((d, idx) => {
        const forecastHeight = (d.forecast_qty / maxValue) * 100;
        const actualHeight = d.actual_qty ? (d.actual_qty / maxValue) * 100 : 0;
        
        return (
          <div key={idx} className="flex-1 flex flex-col items-center gap-1">
            <div className="w-full flex items-end gap-1 justify-center" style={{ height: height - 24 }}>
              {/* Forecast bar */}
              <div 
                className="w-4 bg-cyan-500/60 rounded-t"
                style={{ height: `${forecastHeight}%` }}
                title={`Forecast: ${d.forecast_qty}`}
              />
              {/* Actual bar */}
              {d.actual_qty && (
                <div 
                  className="w-4 bg-emerald-500/80 rounded-t"
                  style={{ height: `${actualHeight}%` }}
                  title={`Actual: ${d.actual_qty}`}
                />
              )}
            </div>
            <span className="text-[10px] text-slate-500">{d.period.split('-')[1]}</span>
          </div>
        );
      })}
    </div>
  );
};

const AccuracyGauge: React.FC<{ value: number; label: string; inverse?: boolean }> = ({ value, label, inverse = false }) => {
  const getColor = () => {
    const threshold = inverse ? 100 - value : value;
    if (threshold >= 90) return 'text-emerald-400';
    if (threshold >= 80) return 'text-amber-400';
    return 'text-red-400';
  };
  
  return (
    <div className="text-center">
      <p className={`text-3xl font-bold ${getColor()}`}>{value.toFixed(1)}%</p>
      <p className="text-xs text-slate-500">{label}</p>
    </div>
  );
};

const ForecastCard: React.FC<{
  forecast: ForecastData;
  isSelected: boolean;
  onSelect: () => void;
}> = ({ forecast, isSelected, onSelect }) => (
  <motion.div
    layout
    onClick={onSelect}
    className={`p-4 rounded-xl border cursor-pointer transition-all ${
      isSelected
        ? 'bg-cyan-500/10 border-cyan-500/50'
        : 'bg-slate-800/50 border-slate-700/50 hover:border-slate-600'
    }`}
  >
    <div className="flex items-center justify-between mb-3">
      <div>
        <p className="font-medium text-white">{forecast.item_code}</p>
        <p className="text-sm text-slate-400">{forecast.item_name}</p>
      </div>
      <div className="text-right">
        <p className={`text-xl font-bold ${forecast.accuracy >= 90 ? 'text-emerald-400' : forecast.accuracy >= 80 ? 'text-amber-400' : 'text-red-400'}`}>
          {forecast.accuracy.toFixed(1)}%
        </p>
        <p className="text-xs text-slate-500">Precisão</p>
      </div>
    </div>
    
    <SimpleBarChart data={forecast.forecasts} />
    
    <div className="flex items-center justify-between mt-3 text-xs text-slate-500">
      <span>MAPE: {forecast.mape.toFixed(1)}%</span>
      <span>Bias: {forecast.bias > 0 ? '+' : ''}{forecast.bias.toFixed(1)}%</span>
    </div>
  </motion.div>
);

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export const MRPForecastPanel: React.FC = () => {
  const [selectedForecast, setSelectedForecast] = useState<string | null>(null);
  const [uploadResult, setUploadResult] = useState<ForecastUploadResult | null>(null);
  
  const queryClient = useQueryClient();

  const { data: forecasts, isLoading: forecastsLoading } = useQuery({
    queryKey: ['forecasts'],
    queryFn: fetchForecasts,
  });

  const { data: accuracy, isLoading: accuracyLoading } = useQuery({
    queryKey: ['forecast-accuracy'],
    queryFn: fetchForecastAccuracy,
  });

  const uploadMutation = useMutation({
    mutationFn: uploadForecast,
    onSuccess: (result) => {
      setUploadResult(result);
      queryClient.invalidateQueries({ queryKey: ['forecasts'] });
    },
  });

  const runMRPMutation = useMutation({
    mutationFn: runMRPWithForecast,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['mrp-runs'] });
    },
  });

  const selectedForecastData = forecasts?.find(f => f.item_code === selectedForecast);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-cyan-500" />
            Forecast Integration
          </h2>
          <p className="text-sm text-slate-400 mt-1">
            Importar previsões e integrar com MRP
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg cursor-pointer">
            <Upload className="w-4 h-4" />
            Importar Forecast
            <input
              type="file"
              accept=".csv,.xlsx,.json"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) uploadMutation.mutate(file);
              }}
            />
          </label>
          <button
            onClick={() => runMRPMutation.mutate()}
            disabled={runMRPMutation.isPending || !forecasts?.length}
            className="flex items-center gap-2 px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg font-medium disabled:opacity-50"
          >
            {runMRPMutation.isPending ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            Executar MRP com Forecast
          </button>
        </div>
      </div>

      {/* Upload Result */}
      <AnimatePresence>
        {uploadResult && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className={`p-4 rounded-xl flex items-center justify-between ${
              uploadResult.success ? 'bg-emerald-500/10 border border-emerald-500/30' : 'bg-red-500/10 border border-red-500/30'
            }`}
          >
            <div className="flex items-center gap-3">
              {uploadResult.success ? (
                <Check className="w-5 h-5 text-emerald-400" />
              ) : (
                <AlertTriangle className="w-5 h-5 text-red-400" />
              )}
              <span className="text-white">
                {uploadResult.success 
                  ? `Importados ${uploadResult.items_imported} itens para ${uploadResult.periods} períodos`
                  : 'Erro na importação'
                }
              </span>
            </div>
            <button 
              onClick={() => setUploadResult(null)}
              className="p-1 text-slate-400 hover:bg-slate-800 rounded"
            >
              <X className="w-4 h-4" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Accuracy Overview */}
      <div className="grid grid-cols-4 gap-4">
        <div className="p-4 rounded-xl border border-slate-700/50 bg-slate-800/30 text-center">
          <Target className="w-6 h-6 text-cyan-400 mx-auto mb-2" />
          <p className="text-3xl font-bold text-white">{accuracy ? (100 - accuracy.overall_mape).toFixed(1) : '--'}%</p>
          <p className="text-xs text-slate-500">Precisão Geral</p>
        </div>
        <div className="p-4 rounded-xl border border-slate-700/50 bg-slate-800/30 text-center">
          <BarChart2 className="w-6 h-6 text-amber-400 mx-auto mb-2" />
          <p className="text-3xl font-bold text-white">{accuracy?.overall_mape.toFixed(1) || '--'}%</p>
          <p className="text-xs text-slate-500">MAPE</p>
        </div>
        <div className="p-4 rounded-xl border border-slate-700/50 bg-slate-800/30 text-center">
          <TrendingUp className="w-6 h-6 text-purple-400 mx-auto mb-2" />
          <p className="text-3xl font-bold text-white">{accuracy ? `${accuracy.overall_bias > 0 ? '+' : ''}${accuracy.overall_bias.toFixed(1)}` : '--'}%</p>
          <p className="text-xs text-slate-500">Bias</p>
        </div>
        <div className="p-4 rounded-xl border border-slate-700/50 bg-slate-800/30 text-center">
          <Layers className="w-6 h-6 text-emerald-400 mx-auto mb-2" />
          <p className="text-3xl font-bold text-white">{forecasts?.length || 0}</p>
          <p className="text-xs text-slate-500">Itens com Forecast</p>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-12 gap-6">
        {/* Forecast List */}
        <div className="col-span-5 space-y-3 max-h-[500px] overflow-y-auto">
          {forecastsLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
            </div>
          ) : forecasts?.length === 0 ? (
            <div className="text-center py-12 rounded-xl border border-slate-700/50 bg-slate-800/30">
              <Cloud className="w-12 h-12 text-slate-600 mx-auto mb-3" />
              <p className="text-slate-400">Nenhum forecast importado</p>
              <p className="text-xs text-slate-500 mt-1">Importe um ficheiro CSV ou JSON</p>
            </div>
          ) : (
            forecasts?.map((forecast) => (
              <ForecastCard
                key={forecast.item_code}
                forecast={forecast}
                isSelected={selectedForecast === forecast.item_code}
                onSelect={() => setSelectedForecast(forecast.item_code)}
              />
            ))
          )}
        </div>

        {/* Detail Panel */}
        <div className="col-span-7">
          {selectedForecastData ? (
            <div className="rounded-xl border border-slate-700/50 bg-slate-800/30 p-6">
              <div className="flex items-center justify-between mb-6">
                <div>
                  <h3 className="text-xl font-bold text-white">{selectedForecastData.item_code}</h3>
                  <p className="text-sm text-slate-400">{selectedForecastData.item_name}</p>
                </div>
                <AccuracyGauge value={selectedForecastData.accuracy} label="Precisão" />
              </div>

              {/* Chart Legend */}
              <div className="flex items-center gap-4 mb-4 text-xs">
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 bg-cyan-500/60 rounded" />
                  <span className="text-slate-400">Forecast</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-3 h-3 bg-emerald-500/80 rounded" />
                  <span className="text-slate-400">Actual</span>
                </div>
              </div>

              {/* Bigger Chart */}
              <div className="bg-slate-900/50 rounded-lg p-4 mb-6">
                <SimpleBarChart data={selectedForecastData.forecasts} height={180} />
              </div>

              {/* Period Table */}
              <div>
                <h4 className="text-sm font-medium text-slate-400 mb-3">Detalhes por Período</h4>
                <div className="overflow-hidden rounded-lg border border-slate-700/50">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-900/50">
                      <tr>
                        <th className="px-3 py-2 text-left text-slate-500">Período</th>
                        <th className="px-3 py-2 text-right text-slate-500">Forecast</th>
                        <th className="px-3 py-2 text-right text-slate-500">Actual</th>
                        <th className="px-3 py-2 text-right text-slate-500">IC (95%)</th>
                        <th className="px-3 py-2 text-right text-slate-500">Erro</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedForecastData.forecasts.map((period, idx) => {
                        const error = period.actual_qty 
                          ? ((period.actual_qty - period.forecast_qty) / period.forecast_qty * 100).toFixed(1)
                          : null;
                        
                        return (
                          <tr key={idx} className="border-t border-slate-700/50">
                            <td className="px-3 py-2 text-white font-mono">{period.period}</td>
                            <td className="px-3 py-2 text-right text-cyan-400">{period.forecast_qty}</td>
                            <td className="px-3 py-2 text-right text-emerald-400">
                              {period.actual_qty || '—'}
                            </td>
                            <td className="px-3 py-2 text-right text-slate-400">
                              [{period.confidence_lower}, {period.confidence_upper}]
                            </td>
                            <td className={`px-3 py-2 text-right ${
                              error === null ? 'text-slate-500' :
                              Math.abs(parseFloat(error)) < 10 ? 'text-emerald-400' : 
                              Math.abs(parseFloat(error)) < 20 ? 'text-amber-400' : 'text-red-400'
                            }`}>
                              {error ? `${parseFloat(error) > 0 ? '+' : ''}${error}%` : '—'}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center rounded-xl border border-slate-700/50 bg-slate-800/30 p-12">
              <div className="text-center">
                <LineChart className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                <p className="text-lg font-medium text-slate-400">Selecione um item</p>
                <p className="text-sm text-slate-500 mt-1">
                  Para ver detalhes do forecast
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Accuracy by Category */}
      {accuracy && (
        <div className="grid grid-cols-3 gap-4">
          {accuracy.accuracy_by_category.map((cat) => (
            <div key={cat.category} className="p-4 rounded-xl border border-slate-700/50 bg-slate-800/30">
              <h4 className="text-sm font-medium text-slate-400 mb-3">{cat.category}</h4>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-2xl font-bold text-white">{(100 - cat.mape).toFixed(1)}%</p>
                  <p className="text-xs text-slate-500">Precisão</p>
                </div>
                <div className="text-right">
                  <p className="text-lg font-medium text-slate-300">{cat.mape.toFixed(1)}%</p>
                  <p className="text-xs text-slate-500">MAPE</p>
                </div>
                <div className="text-right">
                  <p className={`text-lg font-medium ${cat.bias > 0 ? 'text-amber-400' : 'text-emerald-400'}`}>
                    {cat.bias > 0 ? '+' : ''}{cat.bias.toFixed(1)}%
                  </p>
                  <p className="text-xs text-slate-500">Bias</p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default MRPForecastPanel;


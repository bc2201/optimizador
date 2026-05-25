"""
OPTIMIZADOR AUTOMÁTICO MULTI-FASE
=================================
Sistema que converge automáticamente hacia la mejor configuración
mediante 3 fases: Exploración → Refinamiento → Validación.

Uso:
    from optimization import ejecutar_optimizacion_automatica
    
    resultados = ejecutar_optimizacion_automatica(
        df=df,
        config_base=config,
        features=features,
        symbol="BTC/USDT",
        timeframe="1h"
    )
"""

import numpy as np
import threading
from optimizador_main import run_single_optuna, run_backtest


class OptimizadorAutomatico:
    """
    Optimizador que ejecuta automáticamente 3 fases de optimización.
    
    Fase 1: Exploración Rápida (paralelo, pocos trials)
    Fase 2: Refinamiento (serie, rangos acotados, más trials)
    Fase 3: Validación (multi-run, confirma robustez)
    """
    
    def __init__(self, df, config_base, features, symbol, timeframe, 
                 config_fases=None, config_convergencia=None, 
                 config_metricas=None, verbose=True):
        """
        Args:
            df: DataFrame con datos históricos
            config_base: Configuración base de rangos
            features: Features de la estrategia
            symbol: Símbolo del activo (para logs)
            timeframe: Timeframe (para logs)
            config_fases: Configuración de trials y modos (opcional)
            config_convergencia: Configuración de convergencia (opcional)
            config_metricas: Configuración de métricas por fase (opcional)
            verbose: Mostrar logs detallados
        """
        self.df = df
        self.config_base = config_base.copy()
        self.features = features
        self.symbol = symbol
        self.timeframe = timeframe
        self.verbose = verbose
        
        # Cargar configuraciones (usar defaults si no se proveen)
        try:
            from config_auto_optimizer import (CONFIG_FASES, CONFIG_CONVERGENCIA, CONFIG_METRICAS)
            
            self.config_fases = config_fases if config_fases else CONFIG_FASES
            self.config_convergencia = config_convergencia if config_convergencia else CONFIG_CONVERGENCIA
            self.config_metricas = config_metricas if config_metricas else CONFIG_METRICAS
        except ImportError:
            # Si no existe config_auto_optimizer.py, usar valores por defecto
            self.config_fases = config_fases if config_fases else self._default_fases()
            self.config_convergencia = config_convergencia if config_convergencia else self._default_convergencia()
            self.config_metricas = config_metricas if config_metricas else self._default_metricas()
        
        # Historial de resultados
        self.historial = []
    
    def _default_fases(self):
        """Valores por defecto si no existe config_auto_optimizer.py"""
        return {
            "fase_1": {"trials": 2000, "modo": "paralelo"},
            "fase_2": {"trials": 1500, "modo": "serie"},
            "fase_3": {"trials_por_corrida": 800, "corridas": 5, "modo": "serie"}
        }
    
    def _default_convergencia(self):
        """Valores por defecto para convergencia"""
        return {
            "activar": True,
            "ventana": 75,
            "tolerancia": 0.002,
            "trials_minimos": 400,
            "mejor_score_minimo": 0.85
        }
    
    def _default_metricas(self):
        """Valores por defecto para métricas"""
        return {
            "fase_1": {"use_pf": True, "peso_pf": 60.0, "use_winrate": True, "peso_winrate": 40.0,
                      "use_drawdown": False, "peso_drawdown": 0.0, "use_n_trades": False, 
                      "peso_n_trades": 0.0, "min_trades": 15},
            "fase_2": {"use_pf": True, "peso_pf": 40.0, "use_winrate": True, "peso_winrate": 30.0,
                      "use_drawdown": True, "peso_drawdown": 20.0, "use_n_trades": True, 
                      "peso_n_trades": 10.0, "min_trades": 20},
            "fase_3": {"use_pf": True, "peso_pf": 35.0, "use_winrate": True, "peso_winrate": 30.0,
                      "use_drawdown": True, "peso_drawdown": 35.0, "use_n_trades": False, 
                      "peso_n_trades": 0.0, "min_trades": 30}
        }
        
    def _log(self, mensaje):
        """Muestra mensaje si verbose está activado"""
        if self.verbose:
            print(mensaje)
    
    def _acotar_rangos(self, params_referencia):
        """
        Reduce los rangos de búsqueda alrededor de los parámetros encontrados.
        Redondea los valores a 1 decimal para evitar warnings de Optuna.
        """
        config_acotada = self.config_base.copy()
        
        # Acotar MA lengths (±30% alrededor del valor encontrado)
        if "ma1_length" in params_referencia:
            ma1 = params_referencia["ma1_length"]
            config_acotada["ma1_min"] = max(2, int(ma1 * 0.7))
            config_acotada["ma1_max"] = int(ma1 * 1.3)
        
        if "ma2_length" in params_referencia:
            ma2 = params_referencia["ma2_length"]
            config_acotada["ma2_min"] = max(5, int(ma2 * 0.7))
            config_acotada["ma2_max"] = int(ma2 * 1.3)
        
        # Acotar RSI si existe
        if "rsi_length" in params_referencia:
            rsi = params_referencia["rsi_length"]
            rango_actual = self.config_base.get("rsi_length_range", (2, 50))
            nuevo_min = max(2, int(rsi * 0.7))
            nuevo_max = min(rango_actual[1], int(rsi * 1.3))
            if nuevo_min < nuevo_max:
                config_acotada["rsi_length_range"] = (nuevo_min, nuevo_max)
        
        # ============================================================
        # REDONDEAR RANGOS DECIMALES A MÚLTIPLOS DE 0.1
        # ============================================================
        
        # Acotar stop loss (redondear a 1 decimal)
        if "stop_loss_pct" in params_referencia:
            sl = params_referencia["stop_loss_pct"]
            rango_actual = self.config_base.get("sl_range", (0.3, 10.0))
            nuevo_min = round(max(0.3, sl * 0.7), 1)
            nuevo_max = round(min(rango_actual[1], sl * 1.3), 1)
            if nuevo_min < nuevo_max and (nuevo_max - nuevo_min) >= 0.1:
                config_acotada["sl_range"] = (nuevo_min, nuevo_max)
        
        # Acotar take profit long (redondear a 1 decimal)
        if "tp_long_pct" in params_referencia:
            tp = params_referencia["tp_long_pct"]
            rango_actual = self.config_base.get("tp_long_range", (0.3, 99.0))
            nuevo_min = round(max(0.3, tp * 0.7), 1)
            nuevo_max = round(min(rango_actual[1], tp * 1.3), 1)
            if nuevo_min < nuevo_max and (nuevo_max - nuevo_min) >= 0.1:
                config_acotada["tp_long_range"] = (nuevo_min, nuevo_max)
        
        # Acotar take profit short (redondear a 1 decimal)
        if "tp_short_pct" in params_referencia:
            tp = params_referencia["tp_short_pct"]
            rango_actual = self.config_base.get("tp_short_range", (0.3, 99.0))
            nuevo_min = round(max(0.3, tp * 0.7), 1)
            nuevo_max = round(min(rango_actual[1], tp * 1.3), 1)
            if nuevo_min < nuevo_max and (nuevo_max - nuevo_min) >= 0.1:
                config_acotada["tp_short_range"] = (nuevo_min, nuevo_max)
        
        # Acotar ADX threshold (redondear a 1 decimal)
        if "adx_threshold" in params_referencia:
            adx = params_referencia["adx_threshold"]
            rango_actual = self.config_base.get("adx_thr_range", (5.0, 70.0))
            nuevo_min = round(max(5.0, adx * 0.7), 1)
            nuevo_max = round(min(rango_actual[1], adx * 1.3), 1)
            if nuevo_min < nuevo_max and (nuevo_max - nuevo_min) >= 0.1:
                config_acotada["adx_thr_range"] = (nuevo_min, nuevo_max)
        
        # Acotar RSI min (redondear a 1 decimal)
        if "rsi_min" in params_referencia:
            rsi_min = params_referencia["rsi_min"]
            rango_actual = self.config_base.get("rsi_min_range", (50.0, 80.0))
            nuevo_min = round(max(50.0, rsi_min * 0.7), 1)
            nuevo_max = round(min(rango_actual[1], rsi_min * 1.3), 1)
            if nuevo_min < nuevo_max and (nuevo_max - nuevo_min) >= 0.1:
                config_acotada["rsi_min_range"] = (nuevo_min, nuevo_max)
        
        # Acotar RSI max (redondear a 1 decimal)
        if "rsi_max" in params_referencia:
            rsi_max = params_referencia["rsi_max"]
            rango_actual = self.config_base.get("rsi_max_range", (5.0, 50.0))
            nuevo_min = round(max(5.0, rsi_max * 0.7), 1)
            nuevo_max = round(min(rango_actual[1], rsi_max * 1.3), 1)
            if nuevo_min < nuevo_max and (nuevo_max - nuevo_min) >= 0.1:
                config_acotada["rsi_max_range"] = (nuevo_min, nuevo_max)
    
        return config_acotada
    
    def _metrics_config_para_fase(self, fase):
        """Usa la configuración de métricas desde self.config_metricas"""
        return self.config_metricas.get(f"fase_{fase}", {})
    
    def _fase_exploracion_rapida(self):
        """
        FASE 1: Exploración rápida en paralelo con detección de convergencia.
        Objetivo: Descartar malas configuraciones, identificar regiones prometedoras.
        """
        cfg_fase = self.config_fases.get("fase_1", {})
        n_trials = cfg_fase.get("trials", 2000)
        modo_paralelo = cfg_fase.get("modo", "paralelo") == "paralelo"
        
        self._log("\n" + "="*60)
        self._log("🔎 FASE 1: Exploración Rápida")
        
        # Mostrar configuración de convergencia si está activada
        if self.config_convergencia.get("activar", True):
            self._log(f"   Modo: {'Paralelo' if modo_paralelo else 'Serie'} | Trials Max: {n_trials} | Convergencia: Sí")
            self._log(f"   Convergencia: ventana={self.config_convergencia['ventana']}, "
                     f"tolerancia={self.config_convergencia['tolerancia']*100:.1f}%, "
                     f"trials_min={self.config_convergencia['trials_minimos']}")
        else:
            self._log(f"   Modo: {'Paralelo' if modo_paralelo else 'Serie'} | Trials: {n_trials} | Convergencia: No")
        self._log("="*60)
        
        metrics_config = self._metrics_config_para_fase(1)
        
        # ============================================================
        # OPTIMIZACIÓN CON O SIN CONVERGENCIA
        # ============================================================
        
        if self.config_convergencia.get("activar", True):
            # Con detección de convergencia
            import optuna
            
            convergencia_detectada = False
            trial_convergencia = 0
            
            def callback_convergencia(study, trial):
                nonlocal convergencia_detectada, trial_convergencia
                
                # No evaluar antes de trials_minimos
                if trial.number < self.config_convergencia["trials_minimos"]:
                    return
                
                # Evaluar cada 10 trials
                if trial.number % 10 != 0:
                    return
                
                ventana = self.config_convergencia["ventana"]
                if len(study.trials) < ventana:
                    return
                
                # Calcular mejora en la ventana
                ultimos_scores = [t.value for t in study.trials[-ventana:]]
                mejor_en_ventana = max(ultimos_scores)
                peor_en_ventana = min(ultimos_scores)
                
                if peor_en_ventana > 0:
                    mejora_relativa = (mejor_en_ventana - peor_en_ventana) / peor_en_ventana
                else:
                    mejora_relativa = 0
                
                tolerancia = self.config_convergencia["tolerancia"]
                mejor_absoluto = study.best_value
                score_minimo = self.config_convergencia.get("mejor_score_minimo", 0.85)
                
                # Verificar convergencia
                if mejora_relativa < tolerancia or mejor_absoluto > score_minimo:
                    convergencia_detectada = True
                    trial_convergencia = trial.number
                    self._log(f"\n   🎯 Convergencia detectada en trial {trial_convergencia}")
                    self._log(f"      Mejora en últimos {ventana} trials: {mejora_relativa:.3%}")
                    self._log(f"      Mejor score actual: {mejor_absoluto:.4f}")
                    study.stop()
            
            # Crear estudio con callback
            sampler = optuna.samplers.TPESampler(seed=None)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            
            def objective_wrapper(trial):
                from optimizador_main import objective
                return objective(trial, self.df, self.config_base, self.features, 
                                metrics_config, {}, threading.Lock())
            
            study.optimize(
                objective_wrapper,
                n_trials=n_trials,
                n_jobs=-1 if modo_paralelo else 1,
                callbacks=[callback_convergencia]
            )
            
            best_score = study.best_value if study.best_value else 0
            best_params = study.best_params if study.best_params else {}
            
            trials_usados = trial_convergencia if convergencia_detectada else n_trials
            
            if convergencia_detectada:
                self._log(f"\n   ⏹️  Detenido en trial {trials_usados} de {n_trials} (convergencia)")
            else:
                self._log(f"\n   ⏹️  Completados {n_trials} trials (límite alcanzado)")
            
        else:
            # Sin convergencia (método original)
            pf, best_params = run_single_optuna(
                df=self.df,
                config=self.config_base,
                n_trials=n_trials,
                modo_paralelo=modo_paralelo,
                features=self.features,
                metrics_config=metrics_config
            )
            best_score = pf
            trials_usados = n_trials
        
        self.historial.append({
            "fase": 1,
            "nombre": "Exploración Rápida",
            "best_score": best_score,
            "best_params": best_params,
            "trials": trials_usados,
            "trials_max": n_trials,
            "modo": "paralelo" if modo_paralelo else "serie",
            "convergencia_temprana": self.config_convergencia.get("activar", False)
        })
        
        self._log(f"\n   ✅ Mejor score: {best_score:.4f}")
        self._log(f"   📊 Parámetros encontrados: {len(best_params)}")
        
        return best_params
    
    def _fase_refinamiento(self, params_iniciales):
        """
        FASE 2: Refinamiento en serie con rangos acotados.
        Objetivo: Afinar los mejores parámetros.
        """
        cfg_fase = self.config_fases.get("fase_2", {})
        n_trials = cfg_fase.get("trials", 1500)
        modo_paralelo = cfg_fase.get("modo", "serie") == "paralelo"
        
        self._log("\n" + "="*60)
        self._log("🎯 FASE 2: Refinamiento")
        self._log(f"   Modo: {'Paralelo' if modo_paralelo else 'Serie'} | Trials: {n_trials} | Rangos: Acotados")
        self._log("="*60)
        
        # Acotar rangos alrededor de los parámetros encontrados
        config_refinada = self._acotar_rangos(params_iniciales)
        metrics_config = self._metrics_config_para_fase(2)
        
        pf, best_params = run_single_optuna(
            df=self.df,
            config=config_refinada,
            n_trials=n_trials,
            modo_paralelo=modo_paralelo,
            features=self.features,
            metrics_config=metrics_config
        )
        
        self.historial.append({
            "fase": 2,
            "nombre": "Refinamiento",
            "best_score": pf,
            "best_params": best_params,
            "trials": n_trials,
            "modo": "paralelo" if modo_paralelo else "serie",
            "rangos_acotados": True
        })
        
        mejora = (pf - self.historial[0]['best_score']) * 100 if self.historial[0]['best_score'] > 0 else 0
        self._log(f"\n   ✅ Mejor score: {pf:.4f}")
        self._log(f"   📈 Mejora desde fase 1: {mejora:.1f}%")
        
        return best_params
    
    def _fase_validacion(self, params_optimos):
        """
        FASE 3: Validación con multi-run.
        Objetivo: Confirmar robustez y evitar overfitting.
        """
        cfg_fase = self.config_fases.get("fase_3", {})
        trials_por_corrida = cfg_fase.get("trials_por_corrida", 800)
        corridas = cfg_fase.get("corridas", 5)
        modo_paralelo = cfg_fase.get("modo", "serie") == "paralelo"
        
        self._log("\n" + "="*60)
        self._log("🛡️ FASE 3: Validación")
        self._log(f"   Modo: {'Paralelo' if modo_paralelo else 'Serie'} | {corridas} corridas de {trials_por_corrida} trials")
        self._log("="*60)
        
        metrics_config = self._metrics_config_para_fase(3)
        
        resultados_validacion = []
        mejores_params = params_optimos
        mejor_score = 0
        
        for run in range(corridas):
            self._log(f"\n   🔄 Corrida {run+1}/{corridas}...")
            
            pf, params = run_single_optuna(
                df=self.df,
                config=self.config_base,
                n_trials=trials_por_corrida,
                modo_paralelo=modo_paralelo,
                features=self.features,
                metrics_config=metrics_config
            )
            
            resultados_validacion.append(pf)
            
            if pf > mejor_score:
                mejor_score = pf
                mejores_params = params
                self._log(f"      🆕 Nuevo mejor score: {pf:.4f}")
        
        # Estadísticas de validación
        pf_mean = np.mean(resultados_validacion)
        pf_std = np.std(resultados_validacion)
        estabilidad = 1.0 - (pf_std / pf_mean) if pf_mean > 0 else 0
        
        self.historial.append({
            "fase": 3,
            "nombre": "Validación",
            "best_score": pf_mean,
            "best_params": mejores_params,
            "estabilidad": estabilidad,
            "resultados_individuales": resultados_validacion,
            "trials_por_corrida": trials_por_corrida,
            "corridas": corridas,
            "modo": "paralelo" if modo_paralelo else "serie"
        })
        
        self._log(f"\n   ✅ Score promedio: {pf_mean:.4f} (±{pf_std:.4f})")
        self._log(f"   📊 Estabilidad: {estabilidad:.1%}")
        
        return mejores_params
    
    def _generar_reporte_txt(self, best_params, resultado_final):
        """
        Genera un reporte TXT con los mejores parámetros encontrados,
        similar al de la ejecución manual.
        """
        from datetime import datetime
        import os
        
        # Crear carpeta de reportes si no existe
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reportes")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generar timestamp y nombre de archivo
        timestamp = datetime.now().strftime("%Y.%m.%d-%H_%M")
        safe_symbol = self.symbol.replace("/", "_").replace(":", "_")
        filename = f"{timestamp} AutoOptim_{safe_symbol}_{self.timeframe}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # Armar contenido del reporte
        contenido = []
        contenido.append("="*80)
        contenido.append("        REPORTE DE OPTIMIZACIÓN AUTOMÁTICA MULTI-FASE")
        contenido.append("="*80)
        contenido.append("")
        contenido.append(f"📅 Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
        contenido.append(f"📊 Activo: {self.symbol}")
        contenido.append(f"⏱️ Timeframe: {self.timeframe}")
        contenido.append(f"📈 Velas utilizadas: {len(self.df)}")
        contenido.append("")
        
        # Resumen de resultados por fase
        contenido.append("-"*80)
        contenido.append("📋 RESUMEN POR FASE")
        contenido.append("-"*80)
        
        for fase in self.historial:
            contenido.append(f"\n📌 {fase['nombre']} (Fase {fase['fase']})")
            contenido.append(f"   • Mejor score: {fase['best_score']:.4f}")
            contenido.append(f"   • Parámetros: {len(fase.get('best_params', {}))} ajustados")
            if 'estabilidad' in fase:
                contenido.append(f"   • Estabilidad: {fase['estabilidad']:.1%}")
        
        # Parámetros finales encontrados
        contenido.append("")
        contenido.append("-"*80)
        contenido.append("🎯 MEJORES PARÁMETROS ENCONTRADOS")
        contenido.append("-"*80)
        contenido.append("")
        
        # Clasificar parámetros por categoría
        ma_params = {k: v for k, v in best_params.items() if 'ma' in k.lower()}
        rsi_params = {k: v for k, v in best_params.items() if 'rsi' in k.lower()}
        adx_params = {k: v for k, v in best_params.items() if 'adx' in k.lower()}
        risk_params = {k: v for k, v in best_params.items() if any(x in k.lower() for x in ['stop', 'tp', 'be', 'cooldown'])}
        other_params = {k: v for k, v in best_params.items() if k not in ma_params and k not in rsi_params and k not in adx_params and k not in risk_params}
        
        if ma_params:
            contenido.append("📈 MEDIAS MÓVILES:")
            for k, v in ma_params.items():
                contenido.append(f"   • {k}: {v}")
            contenido.append("")
        
        if rsi_params:
            contenido.append("📊 RSI:")
            for k, v in rsi_params.items():
                contenido.append(f"   • {k}: {v}")
            contenido.append("")
        
        if adx_params:
            contenido.append("📉 ADX:")
            for k, v in adx_params.items():
                contenido.append(f"   • {k}: {v}")
            contenido.append("")
        
        if risk_params:
            contenido.append("🛡️ GESTIÓN DE RIESGO:")
            for k, v in risk_params.items():
                contenido.append(f"   • {k}: {v}")
            contenido.append("")
        
        if other_params:
            contenido.append("⚙️ OTROS PARÁMETROS:")
            for k, v in other_params.items():
                contenido.append(f"   • {k}: {v}")
            contenido.append("")
        
        # Estadísticas finales de validación
        contenido.append("-"*80)
        contenido.append("📊 ESTADÍSTICAS DE VALIDACIÓN")
        contenido.append("-"*80)
        contenido.append(f"   • Score promedio: {resultado_final['best_score']:.4f}")
        contenido.append(f"   • Estabilidad: {resultado_final.get('estabilidad', 0):.1%}")
        contenido.append(f"   • Corridas de validación: {len(resultado_final.get('resultados_individuales', []))}")
        
        contenido.append("")
        contenido.append("="*80)
        contenido.append("🔧 Estrategia lista para usar en el optimizador manual")
        contenido.append("   (Puedes cargar estos parámetros en el GUI)")
        contenido.append("="*80)
        
        # Guardar archivo
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(contenido))
        
        print(f"\n📄 Reporte guardado en:\n  {filepath}")

    def _guardar_seed(self, best_params, resultado_final):
        """
        Guarda un archivo JSON (seed) con toda la configuración y resultados,
        similar al que se genera en la ejecución manual.
        """
        from datetime import datetime
        import os
        import json
        
        # Crear carpeta de reportes si no existe
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reportes")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generar timestamp y nombre de archivo
        timestamp = datetime.now().strftime("%Y.%m.%d-%H_%M")
        safe_symbol = self.symbol.replace("/", "_").replace(":", "_")
        filename = f"{timestamp} Seed_AutoOptim_{safe_symbol}_{self.timeframe}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Construir métricas para el seed
        metrics = {
            "profit_factor": resultado_final.get('best_score', 0),
            "estabilidad": resultado_final.get('estabilidad', 0),
            "trades_promedio": None,
            "corridas_validacion": len(resultado_final.get('resultados_individuales', []))
        }
        
        # Construir estructura similar a la seed manual
        seed = {
            "version": "19",
            "timestamp": timestamp,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "tipo_optimizacion": "Automatica (Multi-Fase)",
            
            "configuracion_automatica": {
                "fase_1_trials": self.config_fases.get("fase_1", {}).get("trials", 2000),
                "fase_1_modo": self.config_fases.get("fase_1", {}).get("modo", "paralelo"),
                "fase_2_trials": self.config_fases.get("fase_2", {}).get("trials", 1500),
                "fase_2_modo": self.config_fases.get("fase_2", {}).get("modo", "serie"),
                "fase_3_corridas": self.config_fases.get("fase_3", {}).get("corridas", 5),
                "fase_3_trials_por_corrida": self.config_fases.get("fase_3", {}).get("trials_por_corrida", 800),
                "min_trades_fase_1": self.config_metricas.get("fase_1", {}).get("min_trades", 15),
                "min_trades_fase_2": self.config_metricas.get("fase_2", {}).get("min_trades", 20),
                "min_trades_fase_3": self.config_metricas.get("fase_3", {}).get("min_trades", 30),
                "convergencia_activada": self.config_convergencia.get("activar", True)
            },
            
            "best_params": best_params,
            "metrics": metrics,
            "historial_fases": self.historial,
            
            "output_files": {
                "reporte_txt": None,
                "seed_json": filepath
            }
        }
        
        # Guardar archivo
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(seed, f, indent=4, ensure_ascii=False)
        
        print(f"\n📦 Seed (JSON) guardado en:\n  {filepath}")

    def ejecutar(self):
        """
        Ejecuta el flujo completo de optimización automática.
        
        Returns:
            dict: Resultados finales con mejor score, parámetros y estadísticas
        """
        self._log("\n" + "🚀"*30)
        self._log(" OPTIMIZADOR AUTOMÁTICO MULTI-FASE")
        self._log(f" Activo: {self.symbol} | Timeframe: {self.timeframe}")
        self._log("🚀"*30)
        
        # Fase 1: Exploración
        params_fase1 = self._fase_exploracion_rapida()
        
        # Fase 2: Refinamiento
        params_fase2 = self._fase_refinamiento(params_fase1)
        
        # Fase 3: Validación
        params_fase3 = self._fase_validacion(params_fase2)
        
        # Resultado final
        resultado_final = self.historial[-1]
        
        self._log("\n" + "="*60)
        self._log("✅ OPTIMIZACIÓN AUTOMÁTICA COMPLETADA")
        self._log("="*60)
        self._log(f"\n📊 Mejor score final: {resultado_final['best_score']:.4f}")
        self._log(f"🎯 Estabilidad: {resultado_final.get('estabilidad', 'N/A')}")
        
        # GENERAR REPORTE TXT
        self._generar_reporte_txt(params_fase3, resultado_final)
        
        # GUARDAR SEED JSON
        self._guardar_seed(params_fase3, resultado_final)
        
        return resultado_final
    
    def reporte(self):
        """
        Genera reporte resumido del proceso de optimización.
        """
        print("\n" + "="*70)
        print("  📈 REPORTE DE OPTIMIZACIÓN AUTOMÁTICA")
        print("="*70)
        
        for fase in self.historial:
            print(f"\n📌 {fase['nombre']} (Fase {fase['fase']})")
            print(f"   • Mejor score: {fase['best_score']:.4f}")
            print(f"   • Parámetros: {len(fase.get('best_params', {}))} ajustados")
            if 'estabilidad' in fase:
                print(f"   • Estabilidad: {fase['estabilidad']:.1%}")
        
        print("\n" + "="*70)


def ejecutar_optimizacion_automatica(df, config_base, features, symbol, timeframe, 
                                     config_fases=None, config_convergencia=None,
                                     config_metricas=None, verbose=True):
    """
    Función de alto nivel para ejecutar la optimización automática.
    
    Args:
        df: DataFrame con datos históricos
        config_base: Configuración base de rangos
        features: Features de la estrategia
        symbol: Símbolo del activo
        timeframe: Timeframe
        config_fases: Configuración de trials y modos (opcional)
        config_convergencia: Configuración de convergencia (opcional)
        config_metricas: Configuración de métricas por fase (opcional)
        verbose: Mostrar logs
    
    Returns:
        dict: Resultados finales
    """
    optimizador = OptimizadorAutomatico(
        df=df,
        config_base=config_base,
        features=features,
        symbol=symbol,
        timeframe=timeframe,
        config_fases=config_fases,
        config_convergencia=config_convergencia,
        config_metricas=config_metricas,
        verbose=verbose
    )
    
    resultado = optimizador.ejecutar()
    optimizador.reporte()
    
    return resultado
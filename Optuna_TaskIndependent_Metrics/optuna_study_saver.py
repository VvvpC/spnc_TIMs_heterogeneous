"""
Optuna Study数据保存和载入工具
===================================

本模块提供保存和载入已完成的Optuna Study数据的功能，支持多种格式，
便于后续的数据分析和可视化。

支持功能:
- 保存完整的study数据（trial历史、Pareto前沿、元数据）
- 支持JSON、pickle、CSV等多种格式
- 载入保存的数据进行后续分析
- 数据筛选和查询功能

Author: Chen
Date: 2025-01-XX
"""

import os
import json
import pickle
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import optuna
from optuna.trial import TrialState
import numpy as np


class OptunaStudySaver:
    """Optuna Study数据保存和载入工具类"""
    
    def __init__(self, save_dir: str = "optuna_study_data"):
        """
        初始化保存器
        
        Parameters:
        -----------
        save_dir : str
            数据保存目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_study_data(self, 
                       study: optuna.Study, 
                       filename_prefix: Optional[str] = None,
                       save_formats: List[str] = ["json", "pickle", "csv"]) -> Dict[str, str]:
        """
        保存study的完整数据
        
        Parameters:
        -----------
        study : optuna.Study
            要保存的study对象
        filename_prefix : str, optional
            文件名前缀，默认使用study名称
        save_formats : List[str]
            保存格式列表，可选: ["json", "pickle", "csv"]
            
        Returns:
        --------
        Dict[str, str]: 保存的文件路径字典
        """
        if filename_prefix is None:
            filename_prefix = study.study_name
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_prefix}_{timestamp}"
        
        # 提取study数据
        study_data = self._extract_study_data(study)
        
        saved_files = {}
        
        # 保存为不同格式
        if "json" in save_formats:
            json_path = self._save_as_json(study_data, base_filename)
            saved_files["json"] = json_path
            
        if "pickle" in save_formats:
            pickle_path = self._save_as_pickle(study_data, base_filename)
            saved_files["pickle"] = pickle_path
            
        if "csv" in save_formats:
            csv_paths = self._save_as_csv(study_data, base_filename)
            saved_files.update(csv_paths)
        
        print(f"Study data saved successfully:")
        for format_type, path in saved_files.items():
            print(f"  {format_type}: {path}")
            
        return saved_files
    
    def _extract_study_data(self, study: optuna.Study) -> Dict[str, Any]:
        """提取study的所有相关数据"""
        
        # 基本信息
        study_info = {
            "study_name": study.study_name,
            "directions": [d.name for d in study.directions],
            "n_objectives": len(study.directions),
            "sampler_name": study.sampler.__class__.__name__,
            "creation_time": datetime.now().isoformat(),
            "n_trials": len(study.trials),
            "n_completed_trials": len([t for t in study.trials if t.state == TrialState.COMPLETE])
        }
        
        # Trial数据
        trials_data = []
        for trial in study.trials:
            trial_dict = {
                "number": trial.number,
                "state": trial.state.name,
                "values": trial.values,
                "params": dict(trial.params),
                "user_attrs": dict(trial.user_attrs),
                "system_attrs": dict(trial.system_attrs),
                "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
                "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                "duration": trial.duration.total_seconds() if trial.duration else None
            }
            trials_data.append(trial_dict)
        
        # Pareto前沿数据
        pareto_trials = []
        if hasattr(study, 'best_trials'):
            for trial in study.best_trials:
                pareto_dict = {
                    "number": trial.number,
                    "values": trial.values,
                    "params": dict(trial.params),
                    "user_attrs": dict(trial.user_attrs)
                }
                pareto_trials.append(pareto_dict)
        
        # 汇总数据
        study_data = {
            "study_info": study_info,
            "trials": trials_data,
            "pareto_front": pareto_trials,
            "metadata": {
                "save_timestamp": datetime.now().isoformat(),
                "optuna_version": optuna.__version__
            }
        }
        
        return study_data
    
    def _save_as_json(self, study_data: Dict[str, Any], base_filename: str) -> str:
        """保存为JSON格式"""
        filepath = os.path.join(self.save_dir, f"{base_filename}.json")
        
        # 处理numpy类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj
        
        # 递归转换numpy类型
        def deep_convert(data):
            if isinstance(data, dict):
                return {k: deep_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [deep_convert(item) for item in data]
            else:
                return convert_numpy(data)
        
        converted_data = deep_convert(study_data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def _save_as_pickle(self, study_data: Dict[str, Any], base_filename: str) -> str:
        """保存为pickle格式"""
        filepath = os.path.join(self.save_dir, f"{base_filename}.pkl")
        
        with open(filepath, 'wb') as f:
            pickle.dump(study_data, f)
        
        return filepath
    
    def _save_as_csv(self, study_data: Dict[str, Any], base_filename: str) -> Dict[str, str]:
        """保存为CSV格式"""
        saved_files = {}
        
        # 保存trial数据
        trials_df = pd.DataFrame(study_data["trials"])
        
        # 分离参数和用户属性为单独的列
        if not trials_df.empty:
            # 展开params
            params_df = pd.json_normalize(trials_df['params'])
            params_df.columns = [f"param_{col}" for col in params_df.columns]
            
            # 展开user_attrs
            user_attrs_df = pd.json_normalize(trials_df['user_attrs'])
            user_attrs_df.columns = [f"attr_{col}" for col in user_attrs_df.columns]
            
            # 合并数据
            trials_expanded = pd.concat([
                trials_df.drop(['params', 'user_attrs', 'system_attrs'], axis=1),
                params_df,
                user_attrs_df
            ], axis=1)
            
            trials_path = os.path.join(self.save_dir, f"{base_filename}_trials.csv")
            trials_expanded.to_csv(trials_path, index=False, encoding='utf-8')
            saved_files["trials_csv"] = trials_path
        
        # 保存Pareto前沿数据
        if study_data["pareto_front"]:
            pareto_df = pd.DataFrame(study_data["pareto_front"])
            
            # 展开参数
            pareto_params_df = pd.json_normalize(pareto_df['params'])
            pareto_params_df.columns = [f"param_{col}" for col in pareto_params_df.columns]
            
            # 展开用户属性
            pareto_attrs_df = pd.json_normalize(pareto_df['user_attrs'])
            pareto_attrs_df.columns = [f"attr_{col}" for col in pareto_attrs_df.columns]
            
            pareto_expanded = pd.concat([
                pareto_df.drop(['params', 'user_attrs'], axis=1),
                pareto_params_df,
                pareto_attrs_df
            ], axis=1)
            
            pareto_path = os.path.join(self.save_dir, f"{base_filename}_pareto.csv")
            pareto_expanded.to_csv(pareto_path, index=False, encoding='utf-8')
            saved_files["pareto_csv"] = pareto_path
        
        # 保存study信息
        info_df = pd.DataFrame([study_data["study_info"]])
        info_path = os.path.join(self.save_dir, f"{base_filename}_info.csv")
        info_df.to_csv(info_path, index=False, encoding='utf-8')
        saved_files["info_csv"] = info_path
        
        return saved_files
    
    def load_study_data(self, 
                       filepath: str, 
                       format_type: str = "auto") -> Dict[str, Any]:
        """
        载入保存的study数据
        
        Parameters:
        -----------
        filepath : str
            数据文件路径
        format_type : str
            格式类型，可选: "auto", "json", "pickle"
            
        Returns:
        --------
        Dict[str, Any]: study数据
        """
        if format_type == "auto":
            if filepath.endswith('.json'):
                format_type = "json"
            elif filepath.endswith('.pkl'):
                format_type = "pickle"
            else:
                raise ValueError("Cannot auto-detect format. Please specify format_type.")
        
        if format_type == "json":
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif format_type == "pickle":
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def get_trials_dataframe(self, study_data: Dict[str, Any]) -> pd.DataFrame:
        """将trial数据转换为DataFrame便于分析"""
        trials_df = pd.DataFrame(study_data["trials"])
        
        if not trials_df.empty:
            # 展开参数和属性
            params_df = pd.json_normalize(trials_df['params'])
            params_df.columns = [f"param_{col}" for col in params_df.columns]
            
            user_attrs_df = pd.json_normalize(trials_df['user_attrs'])
            user_attrs_df.columns = [f"attr_{col}" for col in user_attrs_df.columns]
            
            # 合并数据
            expanded_df = pd.concat([
                trials_df.drop(['params', 'user_attrs', 'system_attrs'], axis=1),
                params_df,
                user_attrs_df
            ], axis=1)
            
            return expanded_df
        
        return trials_df
    
    def get_pareto_dataframe(self, study_data: Dict[str, Any]) -> pd.DataFrame:
        """将Pareto前沿数据转换为DataFrame"""
        if not study_data["pareto_front"]:
            return pd.DataFrame()
        
        pareto_df = pd.DataFrame(study_data["pareto_front"])
        
        # 展开参数和属性
        params_df = pd.json_normalize(pareto_df['params'])
        params_df.columns = [f"param_{col}" for col in params_df.columns]
        
        user_attrs_df = pd.json_normalize(pareto_df['user_attrs'])
        user_attrs_df.columns = [f"attr_{col}" for col in user_attrs_df.columns]
        
        expanded_df = pd.concat([
            pareto_df.drop(['params', 'user_attrs'], axis=1),
            params_df,
            user_attrs_df
        ], axis=1)
        
        return expanded_df
    
    def filter_trials(self, 
                     study_data: Dict[str, Any], 
                     state: Optional[str] = "COMPLETE",
                     morph_type: Optional[str] = None,
                     min_cq: Optional[float] = None,
                     min_mc: Optional[float] = None) -> pd.DataFrame:
        """
        筛选trial数据
        
        Parameters:
        -----------
        study_data : Dict[str, Any]
            study数据
        state : str, optional
            trial状态筛选
        morph_type : str, optional
            形貌类型筛选
        min_cq : float, optional
            最小CQ值筛选
        min_mc : float, optional
            最小MC值筛选
            
        Returns:
        --------
        pd.DataFrame: 筛选后的数据
        """
        df = self.get_trials_dataframe(study_data)
        
        if df.empty:
            return df
        
        # 状态筛选
        if state:
            df = df[df['state'] == state]
        
        # 形貌类型筛选
        if morph_type and 'attr_morph_type' in df.columns:
            df = df[df['attr_morph_type'] == morph_type]
        
        # CQ筛选
        if min_cq is not None and 'values' in df.columns:
            df = df[df['values'].apply(lambda x: x[0] >= min_cq if isinstance(x, list) and len(x) > 0 else False)]
        
        # MC筛选
        if min_mc is not None and 'values' in df.columns:
            df = df[df['values'].apply(lambda x: x[1] >= min_mc if isinstance(x, list) and len(x) > 1 else False)]
        
        return df
    
    def print_summary(self, study_data: Dict[str, Any]):
        """打印study数据摘要"""
        info = study_data["study_info"]
        
        print(f"Study Summary:")
        print(f"  Name: {info['study_name']}")
        print(f"  Directions: {info['directions']}")
        print(f"  Sampler: {info['sampler_name']}")
        print(f"  Total trials: {info['n_trials']}")
        print(f"  Completed trials: {info['n_completed_trials']}")
        print(f"  Pareto solutions: {len(study_data['pareto_front'])}")
        
        if study_data["pareto_front"]:
            print(f"\nPareto Front Values:")
            for i, trial in enumerate(study_data["pareto_front"]):
                print(f"  {i+1}. CQ={trial['values'][0]:.4f}, MC={trial['values'][1]:.4f}")


# 便捷函数
def save_study(study: optuna.Study, 
               save_dir: str = "optuna_study_data",
               filename_prefix: Optional[str] = None,
               formats: List[str] = ["json", "pickle", "csv"]) -> Dict[str, str]:
    """
    便捷函数：保存study数据
    
    Parameters:
    -----------
    study : optuna.Study
        要保存的study
    save_dir : str
        保存目录
    filename_prefix : str, optional
        文件名前缀
    formats : List[str]
        保存格式
        
    Returns:
    --------
    Dict[str, str]: 保存的文件路径
    """
    saver = OptunaStudySaver(save_dir)
    return saver.save_study_data(study, filename_prefix, formats)


def load_study(filepath: str, format_type: str = "auto") -> Dict[str, Any]:
    """
    便捷函数：载入study数据
    
    Parameters:
    -----------
    filepath : str
        文件路径
    format_type : str
        格式类型
        
    Returns:
    --------
    Dict[str, Any]: study数据
    """
    saver = OptunaStudySaver()
    return saver.load_study_data(filepath, format_type)
#!/usr/bin/env python3
"""
Dashboard Gradio para reconhecimento de silhueta humana.
6 Tabs: Métricas Gerais, Por Fold, Hiperparâmetros, Detecção, Análise Visual, Config
"""

import gradio as gr
import cv2
import numpy as np
import pandas as pd
import json
import time
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.feature_extractor import LBPFeatureExtractor
from src.config import MODELS_DIR, TARGET_SIZE, YOOSEE_CONFIG
from src.yoosee_camera import YooseeCamera

REPORTS_DIR = Path("reports")


class HumanDetector:
    def __init__(self):
        self.model = None
        self.extractor = None
        self.load_model()
    
    def load_model(self) -> bool:
        models = list(MODELS_DIR.glob("*.pkl"))
        if not models:
            return False
        
        best_models = [m for m in models if "best_model" in m.name]
        if best_models:
            model_path = max(best_models, key=lambda p: p.stat().st_mtime)
        else:
            model_path = max(models, key=lambda p: p.stat().st_mtime)
        
        try:
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict) and "model" in model_data:
                self.model = model_data["model"]
            else:
                self.model = model_data
            self.extractor = LBPFeatureExtractor(radius=1, n_points=8, method='uniform')
            return True
        except Exception as e:
            print(f"Erro ao carregar modelo: {e}")
            return False
    
    def preprocess_frame(self, frame):
        if frame is None:
            return None
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, TARGET_SIZE)
        frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2GRAY)
        return frame_gray
    
    def detect(self, frame):
        if frame is None or self.model is None or self.extractor is None:
            return "ERRO", 0.0
        features = self.extractor.extract_from_image(frame)
        features = features.reshape(1, -1)
        pred = self.model.predict(features)[0]
        proba = self.model.predict_proba(features)[0]
        confidence = np.max(proba)
        return "HUMANO" if pred == 1 else "NÃO HUMANO", confidence
    
    def apply_filter(self, frame, filter_type):
        if frame is None:
            return None
        if filter_type == 'none':
            return frame
        elif filter_type == 'cartoon':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(frame, 9, 300, 300)
            return cv2.bitwise_and(color, color, mask=edges)
        elif filter_type == 'edges':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        elif filter_type == 'colormap':
            return cv2.applyColorMap(frame, cv2.COLORMAP_OCEAN)
        elif filter_type == 'stylized':
            return cv2.stylization(frame, sigma_s=60, sigma_r=0.6)
        elif filter_type == 'pencil':
            _, pencil = cv2.pencilSketch(frame, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
            return cv2.cvtColor(pencil, cv2.COLOR_GRAY2BGR)
        return frame


def load_reports() -> List[Path]:
    reports = list(REPORTS_DIR.glob("model_comparison_*.json"))
    return sorted(reports, key=lambda p: p.stat().st_mtime, reverse=True)


def load_report_data(report_path: Path) -> Dict:
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_report_choices() -> List[str]:
    reports = load_reports()
    if not reports:
        return ["Nenhum relatório encontrado"]
    return [f"{p.name} ({datetime.fromtimestamp(p.stat().st_mtime).strftime('%d/%m/%Y %H:%M')})" for p in reports]


def get_report_path_from_choice(choice: str) -> Optional[Path]:
    if not choice or "Nenhum" in choice:
        return None
    filename = choice.split(" (")[0]
    return REPORTS_DIR / filename


detector = HumanDetector()
yoosee_camera: Optional[YooseeCamera] = None


def render_tab_metrics_general(report_choice: str) -> Tuple[pd.DataFrame, str, str]:
    report_path = get_report_path_from_choice(report_choice)
    if not report_path or not report_path.exists():
        return pd.DataFrame(), "Nenhum relatório encontrado.", ""
    
    data = load_report_data(report_path)
    results = data.get("results", [])
    
    if not results:
        return pd.DataFrame(), "Relatório vazio.", ""
    
    metrics_data = []
    for r in results:
        name = r.get("model_name", "unknown")
        test_m = r.get("test_metrics", {})
        cv_m = r.get("cv_metrics", {})
        cv_std = r.get("cv_std", {})
        
        metrics_data.append({
            "Modelo": name.replace("_", " ").title(),
            "Accuracy (CV)": f"{cv_m.get('accuracy', 0):.4f} ± {cv_std.get('accuracy', 0):.4f}",
            "Accuracy (Test)": f"{test_m.get('accuracy', 0):.4f}",
            "Precision": f"{test_m.get('precision', 0):.4f}",
            "Recall": f"{test_m.get('recall', 0):.4f}",
            "F1-Score": f"{test_m.get('f1_score', 0):.4f}",
            "Tempo (s)": f"{test_m.get('training_time', 0):.2f}"
        })
    
    df = pd.DataFrame(metrics_data)
    
    ranking = data.get("ranking", {})
    best_model = ranking.get("test_accuracy", ["N/A"])[0] if ranking else "N/A"
    
    csv_content = df.to_csv(index=False)
    
    return df, f"Melhor modelo por Accuracy: **{best_model.replace('_', ' ').title()}**", csv_content


def render_tab_metrics_fold(report_choice: str, model_choice: str) -> Tuple[pd.DataFrame, gr.BarPlot]:
    report_path = get_report_path_from_choice(report_choice)
    if not report_path or not report_path.exists():
        return pd.DataFrame(), None
    
    data = load_report_data(report_path)
    results = data.get("results", [])
    
    if not results:
        return pd.DataFrame(), None
    
    model_names = [r.get("model_name", "unknown") for r in results]
    
    if model_choice not in model_names:
        return pd.DataFrame(), None
    
    model_data = next((r for r in results if r.get("model_name") == model_choice), None)
    if not model_data:
        return pd.DataFrame(), None
    
    fold_metrics = model_data.get("cv_fold_metrics", {})
    
    if not fold_metrics or not fold_metrics.get("accuracy"):
        return pd.DataFrame(), None
    
    n_folds = len(fold_metrics["accuracy"])
    fold_data = []
    for i in range(n_folds):
        fold_data.append({
            "Fold": i + 1,
            "Accuracy": fold_metrics["accuracy"][i] if i < len(fold_metrics["accuracy"]) else 0,
            "Precision": fold_metrics["precision"][i] if i < len(fold_metrics["precision"]) else 0,
            "Recall": fold_metrics["recall"][i] if i < len(fold_metrics["recall"]) else 0,
            "F1-Score": fold_metrics["f1_score"][i] if i < len(fold_metrics["f1_score"]) else 0,
        })
    
    avg_row = {
        "Fold": "Média",
        "Accuracy": np.mean(fold_metrics["accuracy"]),
        "Precision": np.mean(fold_metrics["precision"]),
        "Recall": np.mean(fold_metrics["recall"]),
        "F1-Score": np.mean(fold_metrics["f1_score"]),
    }
    
    std_row = {
        "Fold": "Std",
        "Accuracy": np.std(fold_metrics["accuracy"]),
        "Precision": np.std(fold_metrics["precision"]),
        "Recall": np.std(fold_metrics["recall"]),
        "F1-Score": np.std(fold_metrics["f1_score"]),
    }
    
    df = pd.DataFrame(fold_data)
    df_summary = pd.DataFrame([avg_row, std_row])
    df_full = pd.concat([df, df_summary], ignore_index=True)
    
    chart_df = pd.DataFrame({
        'Fold': range(1, n_folds + 1),
        'Accuracy': fold_metrics["accuracy"],
        'F1-Score': fold_metrics["f1_score"]
    })
    
    return df_full, chart_df


def get_model_choices(report_choice: str) -> List[str]:
    report_path = get_report_path_from_choice(report_choice)
    if not report_path or not report_path.exists():
        return []
    
    data = load_report_data(report_path)
    results = data.get("results", [])
    return [r.get("model_name", "unknown") for r in results]


def render_tab_hyperparams(report_choice: str, model_detail: str) -> Tuple[pd.DataFrame, str, str]:
    report_path = get_report_path_from_choice(report_choice)
    if not report_path or not report_path.exists():
        return pd.DataFrame(), "", ""
    
    data = load_report_data(report_path)
    results = data.get("results", [])
    ranking = data.get("ranking", {})
    
    if not results:
        return pd.DataFrame(), "", ""
    
    best_model_name = ranking.get("test_accuracy", [""])[0] if ranking else ""
    
    table_data = []
    for r in results:
        name = r.get("model_name", "unknown")
        best_params = r.get("best_params", {})
        cv_m = r.get("cv_metrics", {})
        cv_std = r.get("cv_std", {})
        val_m = r.get("val_metrics", {})
        test_m = r.get("test_metrics", {})
        
        params_str = ", ".join([f"{k}={v}" for k, v in best_params.items()]) if best_params else "N/A"
        if len(params_str) > 60:
            params_str = params_str[:57] + "..."
        
        is_best = "🏆" if name == best_model_name else ""
        
        table_data.append({
            "": is_best,
            "Modelo": name.replace("_", " ").title(),
            "Hiperparâmetros": params_str,
            "CV Acc": f"{cv_m.get('accuracy', 0):.4f} ± {cv_std.get('accuracy', 0):.4f}",
            "Val Acc": f"{val_m.get('accuracy', 0):.4f}",
            "Test Acc": f"{test_m.get('accuracy', 0):.4f}",
            "Test F1": f"{test_m.get('test_metrics', {}).get('f1_score', 0):.4f}",
        })
    
    df = pd.DataFrame(table_data)
    
    model_data = next((r for r in results if r.get("model_name") == model_detail), None)
    params_json = ""
    if model_data:
        best_params = model_data.get("best_params", {})
        if best_params:
            params_json = json.dumps(best_params, indent=2, ensure_ascii=False)
    
    export_data = []
    for r in results:
        export_data.append({
            "model_name": r.get("model_name", "unknown"),
            "model_type": r.get("model_type", "single"),
            **{f"param_{k}": v for k, v in r.get("best_params", {}).items()},
            "cv_accuracy": r.get("cv_metrics", {}).get("accuracy", 0),
            "cv_std": r.get("cv_std", {}).get("accuracy", 0),
            "val_accuracy": r.get("val_metrics", {}).get("accuracy", 0),
            "val_f1": r.get("val_metrics", {}).get("f1_score", 0),
            "test_accuracy": r.get("test_metrics", {}).get("accuracy", 0),
            "test_f1": r.get("test_metrics", {}).get("f1_score", 0),
            "test_precision": r.get("test_metrics", {}).get("precision", 0),
            "test_recall": r.get("test_metrics", {}).get("recall", 0),
        })
    
    export_df = pd.DataFrame(export_data)
    csv_content = export_df.to_csv(index=False)
    
    return df, params_json, csv_content


def detect_from_webcam(image, filter_type: str) -> Tuple[np.ndarray, str, str]:
    if image is None:
        return np.zeros((256, 256, 3), dtype=np.uint8), "Sem imagem", "0%"
    
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    processed = detector.preprocess_frame(frame)
    label, conf = detector.detect(processed)
    
    filtered = detector.apply_filter(frame, filter_type)
    
    color = (0, 255, 0) if label == "HUMANO" else (0, 0, 255)
    cv2.putText(filtered, f"{label} ({conf:.1%})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    result_rgb = cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
    
    return result_rgb, label, f"{conf:.1%}"


def render_tab_analysis(report_choice: str) -> Tuple[pd.DataFrame, str]:
    report_path = get_report_path_from_choice(report_choice)
    if not report_path or not report_path.exists():
        return pd.DataFrame(), "Nenhum relatório encontrado."
    
    data = load_report_data(report_path)
    results = data.get("results", [])
    
    if not results:
        return pd.DataFrame(), "Relatório vazio."
    
    chart_data = []
    for r in results:
        name = r.get("model_name", "unknown")
        test_m = r.get("test_metrics", {})
        chart_data.append({
            "Modelo": name.replace("_", " ").title(),
            "Accuracy": test_m.get("accuracy", 0),
            "F1-Score": test_m.get("f1_score", 0),
        })
    
    df = pd.DataFrame(chart_data)
    
    ranking = data.get("ranking", {})
    ranking_text = "## Ranking de Modelos\n\n"
    for metric, models in ranking.items():
        ranking_text += f"### {metric}\n"
        for i, m in enumerate(models, 1):
            ranking_text += f"{i}. {m.replace('_', ' ').title()}\n"
        ranking_text += "\n"
    
    return df, ranking_text


def get_dataset_stats() -> Tuple[int, int]:
    data_dir = Path("data/raw")
    human = len(list((data_dir / "human").glob("*.png"))) if (data_dir / "human").exists() else 0
    no_human = len(list((data_dir / "no_human").glob("*.png"))) if (data_dir / "no_human").exists() else 0
    return human, no_human


def build_interface():
    with gr.Blocks(title="Human Recognition Dashboard") as demo:
        gr.Markdown("# Human Recognition Dashboard")
        gr.Markdown("Sistema de reconhecimento de silhueta humana usando LBP + Machine Learning")
        
        with gr.Tabs():
            with gr.TabItem("Métricas Gerais"):
                gr.Markdown("## Métricas Gerais dos Modelos")
                
                with gr.Row():
                    report_dropdown_1 = gr.Dropdown(
                        label="Selecione o relatório",
                        choices=get_report_choices(),
                        value=get_report_choices()[0] if get_report_choices() else None,
                        interactive=True
                    )
                    refresh_btn_1 = gr.Button("Atualizar", variant="secondary")
                
                metrics_table = gr.Dataframe(label="Resumo de Todos os Modelos", interactive=False)
                best_model_text = gr.Markdown()
                
                with gr.Row():
                    csv_output_1 = gr.Textbox(visible=False)
                    download_btn_1 = gr.DownloadButton("Baixar CSV", label="Baixar CSV")
                
                def update_tab1(report):
                    df, text, csv = render_tab_metrics_general(report)
                    return df, text, csv
                
                refresh_btn_1.click(
                    fn=lambda: gr.Dropdown(choices=get_report_choices()),
                    outputs=report_dropdown_1
                )
                
                report_dropdown_1.change(
                    fn=update_tab1,
                    inputs=report_dropdown_1,
                    outputs=[metrics_table, best_model_text, csv_output_1]
                )
                
                download_btn_1.click(
                    fn=lambda csv: csv,
                    inputs=csv_output_1,
                    outputs=download_btn_1
                )
            
            with gr.TabItem("Métricas por Fold"):
                gr.Markdown("## Métricas por Fold")
                
                report_dropdown_2 = gr.Dropdown(
                    label="Relatório",
                    choices=get_report_choices(),
                    value=get_report_choices()[0] if get_report_choices() else None,
                    interactive=True
                )
                
                model_dropdown = gr.Dropdown(
                    label="Selecione o Modelo",
                    choices=[],
                    interactive=True
                )
                
                fold_table = gr.Dataframe(label="Métricas por Fold", interactive=False)
                
                gr.Markdown("### Variação por Fold")
                fold_chart = gr.BarPlot(
                    x="Fold",
                    y="Accuracy",
                    title="Accuracy por Fold",
                    x_title="Fold",
                    y_title="Accuracy"
                )
                
                def update_models(report):
                    models = get_model_choices(report)
                    return gr.Dropdown(choices=models, value=models[0] if models else None)
                
                def update_fold_tab(report, model):
                    df, chart_df = render_tab_metrics_fold(report, model)
                    if chart_df is not None and not chart_df.empty:
                        chart_df_melt = chart_df.melt(id_vars="Fold", var_name="Metric", value_name="Value")
                        return df, chart_df_melt
                    return df, pd.DataFrame()
                
                report_dropdown_2.change(
                    fn=update_models,
                    inputs=report_dropdown_2,
                    outputs=model_dropdown
                )
                
                report_dropdown_2.change(
                    fn=update_fold_tab,
                    inputs=[report_dropdown_2, model_dropdown],
                    outputs=[fold_table, fold_chart]
                )
                
                model_dropdown.change(
                    fn=update_fold_tab,
                    inputs=[report_dropdown_2, model_dropdown],
                    outputs=[fold_table, fold_chart]
                )
            
            with gr.TabItem("Hiperparâmetros"):
                gr.Markdown("## Hiperparâmetros e Métricas")
                
                with gr.Row():
                    report_dropdown_3 = gr.Dropdown(
                        label="Relatório",
                        choices=get_report_choices(),
                        value=get_report_choices()[0] if get_report_choices() else None,
                        interactive=True
                    )
                    refresh_btn_3 = gr.Button("Atualizar", variant="secondary")
                
                hyper_table = gr.Dataframe(label="Tabela Comparativa", interactive=False)
                
                gr.Markdown("### Detalhes dos Hiperparâmetros")
                model_detail_dropdown = gr.Dropdown(
                    label="Selecione o modelo para ver detalhes",
                    choices=[],
                    interactive=True
                )
                params_json = gr.Code(label="Hiperparâmetros Otimizados", language="json", lines=10)
                
                with gr.Row():
                    csv_output_3 = gr.Textbox(visible=False)
                    download_btn_3 = gr.DownloadButton("Baixar CSV", label="Baixar CSV")
                
                def update_models_3(report):
                    models = get_model_choices(report)
                    return gr.Dropdown(choices=models, value=models[0] if models else None)
                
                def update_hyper_tab(report, model):
                    df, params, csv = render_tab_hyperparams(report, model)
                    return df, params, csv
                
                report_dropdown_3.change(
                    fn=update_models_3,
                    inputs=report_dropdown_3,
                    outputs=model_detail_dropdown
                )
                
                report_dropdown_3.change(
                    fn=update_hyper_tab,
                    inputs=[report_dropdown_3, model_detail_dropdown],
                    outputs=[hyper_table, params_json, csv_output_3]
                )
                
                model_detail_dropdown.change(
                    fn=lambda r, m: render_tab_hyperparams(r, m)[1],
                    inputs=[report_dropdown_3, model_detail_dropdown],
                    outputs=params_json
                )
                
                refresh_btn_3.click(
                    fn=lambda: gr.Dropdown(choices=get_report_choices()),
                    outputs=report_dropdown_3
                )
                
                download_btn_3.click(
                    fn=lambda csv: csv,
                    inputs=csv_output_3,
                    outputs=download_btn_3
                )
            
            with gr.TabItem("Detecção"):
                gr.Markdown("## Detecção em Tempo Real")
                
                if detector.model:
                    gr.Success("Modelo carregado com sucesso!")
                else:
                    gr.Warning("Nenhum modelo encontrado. Execute o treinamento primeiro.")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        input_image = gr.Image(label="Entrada", type="numpy")
                        output_image = gr.Image(label="Saída com Detecção")
                    
                    with gr.Column(scale=1):
                        filter_dropdown = gr.Dropdown(
                            label="Filtro",
                            choices=['none', 'cartoon', 'edges', 'colormap', 'stylized', 'pencil'],
                            value='none'
                        )
                        
                        with gr.Group():
                            gr.Markdown("### Resultado")
                            class_output = gr.Textbox(label="Classe", interactive=False)
                            conf_output = gr.Textbox(label="Confiança", interactive=False)
                
                detect_btn = gr.Button("Detectar", variant="primary")
                
                detect_btn.click(
                    fn=detect_from_webcam,
                    inputs=[input_image, filter_dropdown],
                    outputs=[output_image, class_output, conf_output]
                )
            
            with gr.TabItem("Análise Visual"):
                gr.Markdown("## Análise Visual")
                
                report_dropdown_5 = gr.Dropdown(
                    label="Relatório",
                    choices=get_report_choices(),
                    value=get_report_choices()[0] if get_report_choices() else None,
                    interactive=True
                )
                
                gr.Markdown("### Comparação de Accuracy e F1-Score")
                analysis_chart = gr.BarPlot(
                    x="Modelo",
                    y="Accuracy",
                    title="Accuracy por Modelo",
                    x_title="Modelo",
                    y_title="Accuracy"
                )
                
                ranking_markdown = gr.Markdown()
                
                def update_analysis(report):
                    df, text = render_tab_analysis(report)
                    return df, df, text
                
                report_dropdown_5.change(
                    fn=update_analysis,
                    inputs=report_dropdown_5,
                    outputs=[analysis_chart, analysis_chart, ranking_markdown]
                )
            
            with gr.TabItem("Config/Sobre"):
                gr.Markdown("## Configurações")
                
                gr.Markdown("### Câmera Yoosee")
                yoosee_info = f"""
                - **IP:** {YOOSEE_CONFIG.get('ip', 'N/A')}
                - **Porta:** {YOOSEE_CONFIG.get('port', 554)}
                - **Usuário:** {YOOSEE_CONFIG.get('username', 'admin')}
                - **Stream:** {YOOSEE_CONFIG.get('stream', 'onvif1')}
                """
                gr.Markdown(yoosee_info)
                
                gr.Markdown("---")
                gr.Markdown("### Dataset")
                
                human, no_human = get_dataset_stats()
                with gr.Row():
                    human_metric = gr.Number(value=human, label="Humanos")
                    no_human_metric = gr.Number(value=no_human, label="Não Humanos")
                
                gr.Markdown("---")
                gr.Markdown("### Sobre")
                gr.Markdown("""
                **Human Recognition Project**
                
                Sistema de Visão Computacional para reconhecimento de silhueta humana.
                
                - **Features:** LBP (Local Binary Patterns)
                - **Modelos:** Random Forest, XGBoost, SVM, KNN, Logistic Regression, MLP, Gradient Boosting, LightGBM
                - **Validação:** 5-fold Cross-Validation
                - **Deploy:** Hugging Face Spaces (Gradio)
                """)
        
        demo.load(
            fn=update_tab1,
            inputs=report_dropdown_1,
            outputs=[metrics_table, best_model_text, csv_output_1]
        )
        
        demo.load(
            fn=update_models,
            inputs=report_dropdown_2,
            outputs=model_dropdown
        )
        
        demo.load(
            fn=update_models_3,
            inputs=report_dropdown_3,
            outputs=model_detail_dropdown
        )
    
    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        theme=gr.themes.Soft()
    )

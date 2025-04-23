"""
Aplicativo de Análise de Desempenho do Previne Brasil

Este aplicativo apresenta uma análise visual e textual dos indicadores 
do programa Previne Brasil para os municípios selecionados.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import re
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
import traceback


# Carregar variáveis de ambiente do arquivo .env (se existir)
load_dotenv()

# Configuração do logger antes da importação do genai para evitar NameError
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app_desempenho.log'
)
logger = logging.getLogger('app_desempenho')

# Tentativa de importar o Google Generative AI
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
    # Desativar IA se a versão da biblioteca não suportar a interface Client
    if not hasattr(genai, 'Client'):
        HAS_GENAI = False
        logger.warning("Versão incompatível do google-genai instalada. Análise com IA desativada.")
except ImportError:
    HAS_GENAI = False
    logger.warning("Google Generative AI não está disponível. Para instalar: pip install google-genai")

# Carregar variáveis de ambiente
load_dotenv()

# Configurações da página Streamlit
st.set_page_config(
    page_title="Análise Histórica - Previne Brasil",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS simplificado
st.markdown("""
<style>
    .title { text-align: center; font-weight: bold; color: #1E3A8A; }
    .subtitle { text-align: center; color: #1E3A8A; }
    .indicator { font-weight: bold; background-color: #f0f2f6; padding: 5px; border-radius: 5px; }
    .metric-good { color: #0A7B30; font-weight: bold; }
    .metric-medium { color: #FF9800; font-weight: bold; }
    .metric-bad { color: #D32F2F; font-weight: bold; }
    .main .block-container { max-width: 95%; padding-top: 1rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

def carregar_dados_municipios():
    """Carrega dados dos municípios a partir de arquivos JSON"""
    diretorio_atual = Path(__file__).parent.absolute()
    municipios_dados = {}
    
    # Buscar arquivos JSON apenas no diretório atual
    arquivos_json = list(diretorio_atual.glob("*.json"))
    
    if not arquivos_json:
        logger.error("Nenhum arquivo JSON encontrado")
        return {}
    
    # Processar arquivos
    for arquivo in arquivos_json:
        try:
            with open(arquivo, 'r', encoding='utf-8') as f:
                dados = json.load(f)
                
                # Verificar se o arquivo contém dados de um município
                if isinstance(dados, list) and dados and 'Municipio' in dados[0]:
                    municipio_nome = dados[0]['Municipio']
                    municipios_dados[municipio_nome] = dados
                    logger.info(f"Carregado município: {municipio_nome} do arquivo {arquivo.name}")
        except Exception as e:
            logger.error(f"Erro ao processar {arquivo}: {str(e)}")
    
    return municipios_dados

def extrair_periodos(dados):
    """Extrai períodos disponíveis nos dados"""
    periodos = []
    for item in dados[0]:
        if re.match(r'^\d{4}\s+Q\d+\s+\(%\)$', item):
            periodos.append(item)
    return sorted(periodos)

def preparar_dados_indicador(dados, indicador):
    """Prepara dados de um indicador para visualização"""
    for item in dados:
        if item.get('Indicador') == indicador:
            dados_periodos = {'Período': [], 'Valor (%)': [], 'Ano': [], 'Quadrimestre': []}
            
            # Extrair dados de períodos
            for chave, valor in item.items():
                if re.match(r'^\d{4}\s+Q\d+\s+\(%\)$', chave):
                    partes = chave.split()
                    ano = int(partes[0])
                    quadrimestre = partes[1]
                    
                    dados_periodos['Período'].append(chave)
                    try:
                        valor_num = float(valor.replace(',', '.')) if isinstance(valor, str) else float(valor)
                    except (ValueError, TypeError):
                        valor_num = 0
                    
                    dados_periodos['Valor (%)'].append(valor_num)
                    dados_periodos['Ano'].append(ano)
                    dados_periodos['Quadrimestre'].append(quadrimestre)
            
            df = pd.DataFrame(dados_periodos)
            
            # Ordenar cronologicamente
            if not df.empty:
                df['ordem'] = df['Ano']*10 + df['Quadrimestre'].str[1:].astype(int)
                df = df.sort_values('ordem').reset_index(drop=True)
                df.drop('ordem', axis=1, inplace=True)
                df['Ano_Quadrimestre'] = df['Ano'].astype(str) + ' - ' + df['Quadrimestre']
                
            return df
            
    return None

def criar_grafico_linha(df, indicador, meta=None):
    """Cria gráfico de linha para evolução do indicador"""
    if df is None or df.empty:
        return None
    
    # Cores por ano
    cores = {2022: '#1f77b4', 2023: '#ff7f0e', 2024: '#2ca02c', 2025: '#d62728'}
    
    # Criar figura
    fig = go.Figure()
    
    # Adicionar linha de meta
    if meta is not None:
        # Usar coordenadas que funcionem com todos os tipos de eixos x
        fig.add_shape(
            type="line",
            xref="paper",
            x0=0,
            y0=meta,
            x1=1,
            y1=meta,
            line=dict(color="red", width=1, dash="dash"),  # Linha mais grossa
        )
        fig.add_annotation(
            xref="paper",
            x=0.02,
            y=meta,
            text=f"Meta: {meta}%",
            showarrow=False,
            font=dict(color="red", size=10, family="Arial Black"),  # Fonte maior e mais destacada
            bgcolor="rgba(255,255,255,0.8)",
        )
    
    # Plotar linha por ano
    for ano in sorted(df['Ano'].unique()):
        df_ano = df[df['Ano'] == ano]
        if not df_ano.empty:
            fig.add_trace(go.Scatter(
                x=df_ano['Ano_Quadrimestre'],
                y=df_ano['Valor (%)'],
                mode='lines+markers+text',
                name=f'Ano {ano}',
                line=dict(color=cores.get(ano, '#1f77b4'), width=3),
                marker=dict(size=10),
                text=[f"{valor}%" for valor in df_ano['Valor (%)']],
                textposition="top center"
            ))
    
    # Adicionar linha de tendência se houver dados suficientes
    if len(df) >= 3:
        x_indices = list(range(len(df)))
        y_valores = df['Valor (%)'].values
        
        z = np.polyfit(x_indices, y_valores, 1)
        p = np.poly1d(z)
        tendencia_y = p(x_indices)
        
        fig.add_trace(go.Scatter(
            x=df['Ano_Quadrimestre'],
            y=tendencia_y,
            mode='lines',
            name='Tendência',
            line=dict(color='purple', width=2, dash='dot')
        ))
    
    # Verificar se estamos tratando do ISF (que não é percentual)
    is_isf = "Sintético Final (ISF)" in indicador
    
    # Configurar layout
    fig.update_layout(
        title=f'Evolução Histórica: {indicador}',
        title_font=dict(size=18, color="#2C5282"),
        xaxis_title="Período",
        yaxis_title="Valor" if is_isf else "Valor (%)",
        yaxis=dict(range=[0, 10 if is_isf else max(105, df['Valor (%)'].max() + 10)]),
        height=500,
        template="plotly_white",
        showlegend=True  # Garantir que a legenda seja exibida
    )
    
    return fig

def encontrar_meta_para_indicador(indicador):
    """
    Retorna a meta para um indicador baseado em seu nome.
    
    Args:
        indicador: Nome do indicador
    
    Returns:
        float: Valor da meta ou None se não encontrada
    """
    # Metas definidas para cada indicador
    metas = {
        "1.  Proporção de gestantes com pelo menos 6 (seis) consultas pré-natal realizadas, sendo a 1ª (primeira) até a 12ª (décima segunda) semana de gestação": 45,
        "2. Proporção de gestantes com realização de exames para sífilis e HIV": 60,
        "3. Proporção de gestantes com atendimento odontológico realizado": 60,
        "4. Proporção de mulheres com coleta de citopatológico na APS": 40,
        "5.  Proporção de crianças de 1 (um) ano de idade vacinadas na APS contra Difteria, Tétano, Coqueluche, Hepatite B, infecções causadas por haemophilus influenzae tipo b e Poliomielite inativada": 95,
        "6.  Proporção de pessoas com hipertensão, com consulta e pressão arterial aferida no semestre": 50,
        "7.  Proporção de pessoas com diabetes, com consulta e hemoglobina glicada solicitada no semestre": 50,
        "Indicador Sintético Final (ISF)": 7
    }
    
    # Busca direta pela chave exata
    if indicador in metas:
        return metas[indicador]
    
    # Busca por palavras-chave se não encontrar correspondência exata
    palavras_chave = {
        'gestantes': 60,
        'pré-natal': 45,
        'sífilis': 60,
        'hiv': 60,
        'odontológico': 60,
        'citopatológico': 40,
        'mulheres': 40,
        'crianças': 95,
        'vacinadas': 95,
        'hipertensão': 50,
        'diabetes': 50,
        'isf': 7
    }
    
    indicador_lower = indicador.lower()
    for palavra, meta in palavras_chave.items():
        if palavra in indicador_lower:
            return meta
            
    return None

def criar_grafico_radar(dados, ultimo_periodo=None):
    """Cria gráfico radar para comparar indicadores"""
    if not dados:
        return None
    
    # Identificar último período se não fornecido
    if ultimo_periodo is None:
        periodos = set()
        for item in dados:
            for chave in item.keys():
                if re.match(r'^\d{4}\s+Q\d+\s+\(%\)$', chave):
                    periodos.add(chave)
        ultimo_periodo = max(periodos) if periodos else None
    
    if not ultimo_periodo:
        return None
    
    # Coletar dados para o gráfico
    indicadores = []
    valores = []
    metas_valores = []
    
    for item in dados:
        ind = item.get('Indicador', '')
        # Excluir o ISF do gráfico radar
        if ind and ultimo_periodo in item and 'ISF' not in ind:
            # Nome simplificado do indicador
            nome_curto = ind.replace('Proporção de ', '').replace(' realizado', '')
            nome_curto = nome_curto[:40] + '...' if len(nome_curto) > 40 else nome_curto
            
            # Converter o valor para número
            valor_str = item.get(ultimo_periodo, '0')
            try:
                valor = float(valor_str.replace(',', '.')) if isinstance(valor_str, str) else float(valor_str)
            except (ValueError, TypeError):
                valor = 0
                
            indicadores.append(nome_curto)
            valores.append(valor)
            
            # Adicionar a meta específica deste indicador
            meta_valor = encontrar_meta_para_indicador(ind)
            metas_valores.append(meta_valor or 70)  # Usa 70 como fallback
    
    if not indicadores:
        return None
    
    # Criar gráfico radar
    indicadores_plot = indicadores + [indicadores[0]]  # Fechar o polígono
    valores_plot = valores + [valores[0]]
    metas_plot = metas_valores + [metas_valores[0]]  # Fechar o polígono de metas
    
    fig = go.Figure()
    
    # Primeiro: adicionar os valores dos indicadores
    fig.add_trace(go.Scatterpolar(
        r=valores_plot,
        theta=indicadores_plot,
        fill='toself',
        mode='lines+markers+text',
        line_color='blue',
        fillcolor='rgba(0, 128, 255, 0.3)',
        text=[f"{valor:.1f}%" for valor in valores_plot],
        name='Valor Atual'
    ))
    
    # Depois: adicionar a linha de meta
    fig.add_trace(go.Scatterpolar(
        r=metas_plot,
        theta=indicadores_plot,
        fill=None,
        mode='lines',
        line=dict(
            color='red',
            width=3,
            dash='dash'
        ),
        opacity=1.0,
        name='Meta'
    ))
    
    fig.update_layout(
        title=f"Comparativo de Indicadores - {ultimo_periodo}",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig

def criar_grafico_isf(df):
    """Cria gráfico de barras para a evolução do ISF (de 0 a 10)"""
    if df is None or df.empty:
        return None
    
    # Cores para as barras
    paleta_cores = px.colors.qualitative.Safe
    
    # Criar figura de barras com plotly
    fig = go.Figure()
    
    # Adicionar barras para cada período
    fig.add_trace(go.Bar(
        x=df['Ano_Quadrimestre'],
        y=df['Valor (%)'],
        text=[f"{valor:.1f}" for valor in df['Valor (%)']],
        textposition='auto',
        marker_color=[paleta_cores[i % len(paleta_cores)] for i in range(len(df))],
        name='ISF'
    ))
    
    # Adicionar linha de meta se definida (normalmente 7)
    meta = 7  # Meta típica para ISF
    fig.add_shape(
        type="line", 
        x0=-0.5, 
        y0=meta, 
        x1=len(df)-0.5, 
        y1=meta,
        line=dict(color="red", width=3, dash="dash"),
    )
    fig.add_annotation(
        x=0, 
        y=meta+0.3, 
        text=f"Meta: {meta}",
        showarrow=False, 
        font=dict(color="red", size=14, family="Arial Black"),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="red",
        borderwidth=2,
        borderpad=4
    )
    
    # Adicionar linha de tendência se houver dados suficientes
    if len(df) >= 3:
        x_indices = list(range(len(df)))
        y_valores = df['Valor (%)'].values
        
        z = np.polyfit(x_indices, y_valores, 1)
        p = np.poly1d(z)
        tendencia_y = p(x_indices)
        
        fig.add_trace(go.Scatter(
            x=df['Ano_Quadrimestre'],
            y=tendencia_y,
            mode='lines',
            name='Tendência',
            line=dict(color='purple', width=2, dash='dot')
        ))
    
    # Configurar layout
    fig.update_layout(
        title='Evolução do Indicador Sintético Final (ISF)',
        title_font=dict(size=18, color="#2C5282"),
        xaxis_title="Período",
        yaxis_title="Valor ISF (0-10)",
        yaxis=dict(range=[0, 10]),
        height=500,
        template="plotly_white",
        showlegend=True  # Garantir que a legenda seja exibida
    )
    
    return fig

def calcular_estatisticas(df):
    """Calcula estatísticas do indicador"""
    if df is None or df.empty or len(df) < 2:
        return None
    
    estatisticas = {
        'media': np.mean(df['Valor (%)']),
        'mediana': np.median(df['Valor (%)']),
        'minimo': np.min(df['Valor (%)']),
        'maximo': np.max(df['Valor (%)']),
        'ultimo_valor': df['Valor (%)'].iloc[-1],
        'penultimo_valor': df['Valor (%)'].iloc[-2] if len(df) > 1 else None,
        'variacao_ultimo': df['Valor (%)'].iloc[-1] - df['Valor (%)'].iloc[-2] if len(df) > 1 else None,
        'tendencia': 'estável'
    }
    
    # Determinar tendência com base nos últimos 3 valores
    if len(df) >= 3:
        ultimos_valores = df['Valor (%)'].iloc[-3:].tolist()
        diferencas = [ultimos_valores[i+1] - ultimos_valores[i] for i in range(len(ultimos_valores)-1)]
        
        if all(d > 0 for d in diferencas):
            estatisticas['tendencia'] = 'crescimento'
        elif all(d < 0 for d in diferencas):
            estatisticas['tendencia'] = 'queda'
        else:
            estatisticas['tendencia'] = 'oscilação'
    
    return estatisticas

def calcular_isf(dados):
    """
    Calcula o Indicador Sintético Final (ISF) para cada quadrimestre
    usando a metodologia oficial do Previne Brasil com ponderação
    O ISF é um valor de 0 a 10 (não um percentual)
    """
    import math  # Para cálculos de soma mais precisos
    
    # Identificar todos os períodos disponíveis
    periodos = set()
    for item in dados:
        for chave in item.keys():
            if re.match(r'^\d{4}\s+Q\d+\s+\(%\)$', chave):
                periodos.add(chave)
    
    # Ordenar períodos cronologicamente
    periodos = sorted(list(periodos))
    logger.info(f"Períodos identificados: {periodos}")
    
    # Definir metas e ponderações para cada indicador (conforme metodologia oficial)
    config_indicadores = {
        "1.  Proporção de gestantes com pelo menos 6 (seis) consultas pré-natal realizadas, sendo a 1ª (primeira) até a 12ª (décima segunda) semana de gestação": 
            {"meta": 45, "ponderacao": 1},
        "2. Proporção de gestantes com realização de exames para sífilis e HIV": 
            {"meta": 60, "ponderacao": 1},
        "3. Proporção de gestantes com atendimento odontológico realizado": 
            {"meta": 60, "ponderacao": 2},
        "4. Proporção de mulheres com coleta de citopatológico na APS": 
            {"meta": 40, "ponderacao": 1},
        "5.  Proporção de crianças de 1 (um) ano de idade vacinadas na APS contra Difteria, Tétano, Coqueluche, Hepatite B, infecções causadas por haemophilus influenzae tipo b e Poliomielite inativada": 
            {"meta": 95, "ponderacao": 2},
        "6.  Proporção de pessoas com hipertensão, com consulta e pressão arterial aferida no semestre": 
            {"meta": 50, "ponderacao": 2},
        "7.  Proporção de pessoas com diabetes, com consulta e hemoglobina glicada solicitada no semestre": 
            {"meta": 50, "ponderacao": 1}
    }
    
    # Inicializar dicionário para armazenar ISF por período
    isf_calculado = {}
    
    # Função auxiliar para converter texto em número
    def parse_valor(valor_texto):
        if not valor_texto:
            return 0
        
        valor_original = valor_texto
        # Remover caracteres não numéricos, exceto ponto e vírgula
        valor_limpo = ''.join(c for c in str(valor_texto) if c.isdigit() or c in '.,')
        
        # Substituir vírgula por ponto (formato brasileiro para decimal)
        valor_limpo = valor_limpo.replace(',', '.')
        
        try:
            valor_final = float(valor_limpo)
            logger.info(f"Conversão de valor: {valor_original} -> {valor_final}")
            return valor_final
        except (ValueError, TypeError):
            logger.warning(f"Não foi possível converter valor '{valor_texto}' para número")
            return 0
    
    # Calcular ISF para cada período
    for periodo in periodos:
        logger.info(f"\n=== Calculando ISF para {periodo} ===")
        resultados_ponderados = []
        indicadores_processados = []
        
        # Processar cada indicador para este período
        for item in dados:
            if 'Indicador' in item:
                indicador_nome = item['Indicador']
                
                # Verificar se o indicador está na configuração
                config = config_indicadores.get(indicador_nome)
                
                if config and periodo in item:
                    # Obter e converter o resultado para número
                    resultado = parse_valor(item[periodo])
                    meta = config["meta"]
                    ponderacao = config["ponderacao"]
                    
                    # Calcular pontuação (resultado/meta * 10, limitado a 10)
                    if meta > 0:  # Evitar divisão por zero
                        pontuacao = min(10, (resultado / meta) * 10)
                    else:
                        pontuacao = 0
                    
                    # Calcular resultado ponderado
                    resultado_ponderado = pontuacao * ponderacao
                    resultados_ponderados.append(resultado_ponderado)
                    
                    # Registrar o indicador processado para diagnóstico
                    indicadores_processados.append({
                        "nome": indicador_nome,
                        "valor_convertido": resultado,
                        "meta": meta,
                        "ponderacao": ponderacao,
                        "pontuacao": pontuacao,
                        "resultado_ponderado": resultado_ponderado
                    })
        
        # Calcular ISF para este período (soma ponderada / 10)
        if resultados_ponderados:
            soma_ponderada = math.fsum(resultados_ponderados)
            isf_calculado[periodo] = round(soma_ponderada / 10, 1)
            logger.info(f"ISF para {periodo}: {isf_calculado[periodo]}")
        else:
            isf_calculado[periodo] = 0
            logger.warning(f"Nenhum indicador encontrado para calcular ISF no período {periodo}")
    
    # Criar um item no formato dos outros indicadores para o ISF
    isf_item = {
        'Indicador': 'Indicador Sintético Final (ISF)',
        'ISF': True  # Marcador para identificar que este é o ISF
    }
    
    # Adicionar os valores calculados do ISF para cada período
    for periodo, valor in isf_calculado.items():
        isf_item[periodo] = str(valor)
    
    return isf_item

def analisar_com_gemini(dados, periodo, tema=None):
    """Analisa os dados usando a API do Google Gemini."""
    try:
        import os
        import google.generativeai as genai
        
        # Obter a chave API do ambiente
        api_key = os.environ.get("GOOGLE_API_KEY", None)
        if not api_key:
            logger.warning("Chave API do Google Gemini não encontrada nas variáveis de ambiente")
            return "Análise com IA não disponível - configure a chave API do Google Gemini"
        
        # Configure a API
        genai.configure(api_key=api_key)
        
        # Preparar prompt com os dados
        prompt = f"Analise em até dois paragrafos, destacando apenas os pontos positivos, os seguintes indicadores de saúde para o período {periodo}:\n\n"
        if tema:
            prompt += f"Tema solicitado: {tema}\n\n"
        
        prompt += "Indicadores:\n"
        
        # Verificar se dados é um dicionário ou uma lista
        if isinstance(dados, list):
            for indicador in dados:
                if isinstance(indicador, dict) and 'Indicador' in indicador:
                    prompt += f"- {indicador['Indicador']}: {indicador.get(periodo, 'N/A')}\n"
        elif isinstance(dados, dict):
            for chave, valor in dados.items():
                prompt += f"- {chave}: {valor.get(periodo, 'N/A')}\n"
        
        # Usar a nova API para gerar o texto
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        
        # Extrair o texto da resposta
        if hasattr(response, 'text'):
            return response.text
        elif hasattr(response, 'parts'):
            return ''.join(part.text for part in response.parts if hasattr(part, 'text'))
        else:
            return "Análise não gerada - formato de resposta não reconhecido"
    except Exception as e:
        logger.error(f"Erro ao usar a API do Google Gemini: {str(e)}")
        return f"Erro ao usar a API do Google Gemini: {str(e)}"

def gerar_analise_textual(indicador, estatisticas, meta=None):
    """
    Gera uma análise textual para um indicador com base em suas estatísticas.
    
    Args:
        indicador: Nome do indicador
        estatisticas: Dicionário com estatísticas (media, mediana, minimo, maximo, etc.)
        meta: Valor da meta para este indicador
        
    Returns:
        String com a análise textual formatada
    """
    if not estatisticas:
        return "Dados insuficientes para análise."
    
    # Tratamento especial para ISF (escala 0-10)
    is_isf = "Sintético Final (ISF)" in indicador
    
    # Extrair valores estatísticos
    ultimo_valor = estatisticas.get('ultimo_valor', 0)
    penultimo_valor = estatisticas.get('penultimo_valor')
    variacao = estatisticas.get('variacao_ultimo')
    tendencia = estatisticas.get('tendencia', 'estável')
    media = estatisticas.get('media', 0)
    
    # Formatar valores
    formato = ".1f"  # Uma casa decimal
    
    # Iniciar análise
    analise = []
    
    # Cabeçalho
    if is_isf:
        analise.append(f"### Análise do {indicador}")
    else:
        analise.append(f"### Análise do Indicador")
    
    # Comparação com a meta
    if meta is not None:
        if ultimo_valor >= meta:
            if is_isf:
                analise.append(f"🟢 **Meta atingida**: O valor atual é {ultimo_valor:{formato}} (meta: {meta:{formato}}), o que representa **{(ultimo_valor/meta*100):{formato}}%** da meta estabelecida.")
            else:
                analise.append(f"🟢 **Meta atingida**: O valor atual é {ultimo_valor:{formato}}% (meta: {meta:{formato}}%), o que representa **{(ultimo_valor/meta*100):{formato}}%** da meta estabelecida.")
        else:
            if is_isf:
                analise.append(f"🔴 **Meta não atingida**: O valor atual é {ultimo_valor:{formato}} (meta: {meta:{formato}}), o que representa **{(ultimo_valor/meta*100):{formato}}%** da meta estabelecida.")
            else:
                analise.append(f"🔴 **Meta não atingida**: O valor atual é {ultimo_valor:{formato}}% (meta: {meta:{formato}}%), o que representa **{(ultimo_valor/meta*100):{formato}}%** da meta estabelecida.")
    
    # Análise de tendência
    if penultimo_valor is not None and variacao is not None:
        # Formato da variação
        if variacao > 0:
            if is_isf:
                analise.append(f"📈 **Evolução positiva**: Aumento de {abs(variacao):{formato}} pontos em relação ao período anterior ({penultimo_valor:{formato}} → {ultimo_valor:{formato}}).")
            else:
                analise.append(f"📈 **Evolução positiva**: Aumento de {abs(variacao):{formato}} pontos percentuais em relação ao período anterior ({penultimo_valor:{formato}}% → {ultimo_valor:{formato}}%).")
        elif variacao < 0:
            if is_isf:
                analise.append(f"📉 **Evolução negativa**: Redução de {abs(variacao):{formato}} pontos em relação ao período anterior ({penultimo_valor:{formato}} → {ultimo_valor:{formato}}).")
            else:
                analise.append(f"📉 **Evolução negativa**: Redução de {abs(variacao):{formato}} pontos percentuais em relação ao período anterior ({penultimo_valor:{formato}}% → {ultimo_valor:{formato}}%).")
        else:
            if is_isf:
                analise.append(f"➡️ **Estabilidade**: Manteve-se em {ultimo_valor:{formato}} pontos em relação ao período anterior.")
            else:
                analise.append(f"➡️ **Estabilidade**: Manteve-se em {ultimo_valor:{formato}}% em relação ao período anterior.")
    
    # Análise da tendência geral
    if tendencia == 'crescimento':
        analise.append(f"📈 **Tendência geral**: Em crescimento nos últimos períodos.")
    elif tendencia == 'queda':
        analise.append(f"📉 **Tendência geral**: Em queda nos últimos períodos.")
    elif tendencia == 'oscilação':
        analise.append(f"↕️ **Tendência geral**: Com oscilações nos últimos períodos.")
    else:
        analise.append(f"➡️ **Tendência geral**: Relativamente estável.")
    
    # Comparação com média histórica
    if ultimo_valor > media:
        if is_isf:
            analise.append(f"📊 O valor atual de {ultimo_valor:{formato}} está **acima** da média histórica ({media:{formato}}).")
        else:
            analise.append(f"📊 O valor atual de {ultimo_valor:{formato}}% está **acima** da média histórica ({media:{formato}}%).")
    elif ultimo_valor < media:
        if is_isf:
            analise.append(f"📊 O valor atual de {ultimo_valor:{formato}} está **abaixo** da média histórica ({media:{formato}}).")
        else:
            analise.append(f"📊 O valor atual de {ultimo_valor:{formato}}% está **abaixo** da média histórica ({media:{formato}}%).")
    else:
        if is_isf:
            analise.append(f"📊 O valor atual de {ultimo_valor:{formato}} está **igual** à média histórica.")
        else:
            analise.append(f"📊 O valor atual de {ultimo_valor:{formato}}% está **igual** à média histórica.")
    
    # Recomendações específicas para o ISF
    if is_isf:
        if ultimo_valor < 7:
            analise.append("\n#### Recomendações")
            analise.append("- Focar na melhoria dos indicadores com maior peso na composição do ISF")
            analise.append("- Implementar monitoramento frequente dos indicadores críticos")
            analise.append("- Considerar ajustes nas estratégias de saúde para áreas deficitárias")
    # Recomendações para outros indicadores
    elif meta is not None and ultimo_valor < meta:
        analise.append("\n#### Recomendações")
        
        # Recomendações específicas por tipo de indicador
        if "gestantes" in indicador.lower():
            analise.append("- Reforçar a busca ativa de gestantes na área de abrangência")
            analise.append("- Garantir agendamento prioritário para consultas de pré-natal")
            analise.append("- Implementar lembretes para consultas e exames")
        elif "mulheres" in indicador.lower() and "citopatológico" in indicador.lower():
            analise.append("- Aumentar campanhas de conscientização sobre a importância do exame preventivo")
            analise.append("- Facilitar o acesso ao exame com horários estendidos")
            analise.append("- Realizar busca ativa de mulheres na faixa etária alvo")
        elif "crianças" in indicador.lower() and "vacina" in indicador.lower():
            analise.append("- Intensificar campanhas de vacinação nas áreas com menor cobertura")
            analise.append("- Implementar sistema de lembretes para pais/responsáveis")
            analise.append("- Realizar busca ativa de crianças com calendário vacinal atrasado")
        elif "hipertensão" in indicador.lower():
            analise.append("- Aprimorar o acompanhamento dos hipertensos cadastrados")
            analise.append("- Implementar grupos de educação em saúde para hipertensos")
            analise.append("- Facilitar o acesso às consultas e aferição de pressão arterial")
        elif "diabetes" in indicador.lower():
            analise.append("- Intensificar o monitoramento dos pacientes diabéticos")
            analise.append("- Garantir a solicitação rotineira de hemoglobina glicada")
            analise.append("- Promover ações de educação em saúde para pacientes diabéticos")
        else:
            analise.append("- Analisar fatores que impactam negativamente o indicador")
            analise.append("- Implementar estratégias específicas para melhorar o desempenho")
            analise.append("- Monitorar periodicamente os resultados")
    
    return "\n".join(analise)

def mostrar_analise_ia(dados, periodo, container, indicador=None):
    """
    Exibe análise de IA em um container do Streamlit.
    
    Args:
        dados: Dados completos ou filtrados para análise
        periodo: Período a ser analisado (ex: '2023 Q3 (%)')
        container: Container do Streamlit para exibir a análise
        indicador: Nome do indicador específico (opcional)
    """
    with container:
        with st.spinner("Gerando análise com IA..."):
            try:
                # Filtrar dados para o indicador específico se fornecido
                if indicador:
                    dados_filtrados = [item for item in dados if item.get('Indicador') == indicador]
                    if not dados_filtrados:
                        st.warning(f"Não foram encontrados dados para o indicador: {indicador}")
                        return
                    analise = analisar_com_gemini(dados_filtrados, periodo)
                else:
                    analise = analisar_com_gemini(dados, periodo)
                    
                st.markdown(analise)
            except Exception as e:
                st.error(f"Erro ao gerar análise com IA: {str(e)}")
                logger.error(f"Erro na análise IA: {str(e)}\n{traceback.format_exc()}")
                st.info("Continue usando as outras funcionalidades do aplicativo.")

def gerar_dashboard(municipio, dados):
    """Gera o dashboard principal com visualizações"""
    # Calcular o ISF e adicionar aos dados
    isf_item = calcular_isf(dados)
    dados.append(isf_item)
    
    st.markdown(f'<h1 class="title">Análise Histórica - Previne Brasil: {municipio}</h1>', unsafe_allow_html=True)
    
    st.write("""
    ## Sobre o Previne Brasil
    
    O **Previne Brasil** é um programa de financiamento da Atenção Primária à Saúde (APS) que 
    determina o valor dos repasses federais aos municípios com base no desempenho em indicadores
    de saúde. Esta análise apresenta a evolução histórica dos indicadores ao longo do tempo.
    """)
    
    # Informações básicas do município
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Informações do Município")
        if dados and len(dados) > 0:
            st.write(f"**Estado:** {dados[0].get('UF', 'N/A')}")
            st.write(f"**Código IBGE:** {dados[0].get('IBGE', 'N/A')}")
            st.write(f"**Total de indicadores:** {len(dados)}")
    
    # Períodos analisados
    if dados and len(dados) > 0:
        periodos = extrair_periodos(dados)
        with col2:
            st.subheader("Períodos Analisados")
            st.write(f"**Primeiro período:** {periodos[0] if periodos else 'N/A'}")
            st.write(f"**Último período:** {periodos[-1] if periodos else 'N/A'}")
            st.write(f"**Total de períodos:** {len(periodos)}")
    
    # Gráfico radar com panorama geral
    st.subheader("Panorama Geral dos Indicadores")
    radar_fig = criar_grafico_radar(dados)
    if radar_fig:
        st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.error("Dados insuficientes para gerar o panorama geral.")
    
    st.markdown("---")
    
    # Áreas estratégicas de cada indicador
    areas_estrategicas = {
        "1.  Proporção de gestantes com pelo menos 6 (seis) consultas pré-natal realizadas, sendo a 1ª (primeira) até a 12ª (décima segunda) semana de gestação": "Pré-natal",
        "2. Proporção de gestantes com realização de exames para sífilis e HIV": "Pré-natal",
        "3. Proporção de gestantes com atendimento odontológico realizado": "Saúde Bucal",
        "4. Proporção de mulheres com coleta de citopatológico na APS": "Saúde da Mulher",
        "5.  Proporção de crianças de 1 (um) ano de idade vacinadas na APS contra Difteria, Tétano, Coqueluche, Hepatite B, infecções causadas por haemophilus influenzae tipo b e Poliomielite inativada": "Saúde da Criança",
        "6.  Proporção de pessoas com hipertensão, com consulta e pressão arterial aferida no semestre": "Condições Crônicas",
        "7.  Proporção de pessoas com diabetes, com consulta e hemoglobina glicada solicitada no semestre": "Condições Crônicas",
        "Indicador Sintético Final (ISF)": "Indicador Sintético"
    }
    
    # Análise por indicador
    st.header("Análise Histórica dos Indicadores")
    
    modo_visualizacao = st.radio(
        "Escolha o modo de visualização:",
        ["Visualização em abas", "Visualização em lista"],
        horizontal=True
    )
    
    # Extrair lista de indicadores dos dados
    indicadores_unicos = []
    for item in dados:
        if 'Indicador' in item:
            indicador = item['Indicador']
            if indicador not in indicadores_unicos:
                if indicador != "Indicador Sintético Final (ISF)":  # Deixar o ISF para o final
                    indicadores_unicos.append(indicador)
    
    # Adicionar o ISF ao final
    for item in dados:
        if 'Indicador' in item and item['Indicador'] == "Indicador Sintético Final (ISF)":
            indicadores_unicos.append(item['Indicador'])
            break
    
    if modo_visualizacao == "Visualização em abas":
        # Criar abas para cada indicador
        tabs = st.tabs([f"Ind. {i+1}" if indicadores_unicos[i] != "Indicador Sintético Final (ISF)" else "ISF" 
                       for i in range(len(indicadores_unicos))])
        
        for i, tab in enumerate(tabs):
            with tab:
                indicador = indicadores_unicos[i]
                area = areas_estrategicas.get(indicador, "")
                
                # Mostrar número do indicador e área estratégica
                if indicador == "Indicador Sintético Final (ISF)":
                    st.subheader(f"ISF: {indicador}")
                else:
                    st.subheader(f"Indicador {i+1}: {indicador}")
                
                st.write(f"**Área Estratégica:** {area}")
                
                df_indicador = preparar_dados_indicador(dados, indicador)
                
                if df_indicador is not None and not df_indicador.empty:
                    estatisticas = calcular_estatisticas(df_indicador)
                    meta = encontrar_meta_para_indicador(indicador)
                    
                    # Gráfico de evolução - especial para ISF
                    if indicador == "Indicador Sintético Final (ISF)":
                        fig = criar_grafico_isf(df_indicador)
                    else:
                        fig = criar_grafico_linha(df_indicador, indicador, meta)
                        
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Análise textual tradicional
                    analise = gerar_analise_textual(indicador, estatisticas, meta)
                    st.markdown(analise)
                    
                    # Adicionar análise do Gemini com tratamento de erro melhorado
                    with st.expander("🤖 Análise avançada com IA"):
                        # Obter o último período para este indicador
                        ultimo_periodo = periodos[-1] if periodos else None
                        if ultimo_periodo:
                            mostrar_analise_ia(dados, ultimo_periodo, st.container(), indicador)
                        else:
                            st.warning("Período não identificado para análise com IA.")
                else:
                    st.warning("Dados insuficientes para este indicador.")
    else:
        # Visualização em lista
        for i, indicador in enumerate(indicadores_unicos):
            area = areas_estrategicas.get(indicador, "")
            
            # Mostrar número do indicador e área estratégica
            if indicador == "Indicador Sintético Final (ISF)":
                st.subheader(f"ISF: {indicador}")
            else:
                st.subheader(f"{i+1}. {indicador}")
            
            st.write(f"**Área Estratégica:** {area}")
            
            df_indicador = preparar_dados_indicador(dados, indicador)
            
            if df_indicador is not None and not df_indicador.empty:
                estatisticas = calcular_estatisticas(df_indicador)
                meta = encontrar_meta_para_indicador(indicador)
                
                # Resumo rápido
                col1, col2, col3 = st.columns([1, 1, 1])
                
                # Valor atual com indicação visual - tratamento especial para ISF
                valor_atual = estatisticas['ultimo_valor'] if estatisticas else 0
                if indicador == "Indicador Sintético Final (ISF)":
                    # Para ISF, a escala é de 0-10, não percentual
                    if valor_atual >= (meta or 7):
                        col1.markdown(f"<div class='metric-good'>Atual: {valor_atual:.1f}</div>", unsafe_allow_html=True)
                    else:
                        col1.markdown(f"<div class='metric-bad'>Atual: {valor_atual:.1f}</div>", unsafe_allow_html=True)
                    
                    if meta:
                        col2.write(f"**Meta**: {meta:.1f}")
                else:
                    # Para os outros indicadores, continuamos com %
                    if valor_atual >= (meta or 70):
                        col1.markdown(f"<div class='metric-good'>Atual: {valor_atual:.1f}%</div>", unsafe_allow_html=True)
                    else:
                        col1.markdown(f"<div class='metric-bad'>Atual: {valor_atual:.1f}%</div>", unsafe_allow_html=True)
                    
                    if meta:
                        col2.write(f"**Meta**: {meta:.1f}%")
                
                col3.write(f"**Área**: {area}")
                
                # Exibir meta em todos os indicadores
                if meta:
                    st.progress(min(1.0, valor_atual / meta))
                    if indicador == "Indicador Sintético Final (ISF)":
                        st.write(f"**Progresso em relação à meta ({meta:.1f}):** {min(100, valor_atual/meta*100):.1f}%")
                    else:
                        st.write(f"**Progresso em relação à meta ({meta:.1f}%):** {min(100, valor_atual/meta*100):.1f}%")
                
                # Gráfico de evolução - especial para ISF
                if indicador == "Indicador Sintético Final (ISF)":
                    fig = criar_grafico_isf(df_indicador)
                else:
                    fig = criar_grafico_linha(df_indicador, indicador, meta)
                    
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Análise textual
                st.markdown(gerar_analise_textual(indicador, estatisticas, meta))
                
                # Adicionar análise do Gemini com tratamento de erro melhorado
                with st.expander("🤖 Análise avançada com IA"):
                    # Obter o último período para este indicador
                    ultimo_periodo = periodos[-1] if periodos else None
                    if ultimo_periodo:
                        mostrar_analise_ia(dados, ultimo_periodo, st.container(), indicador)
                    else:
                        st.warning("Período não identificado para análise com IA.")
            else:
                st.warning("Dados insuficientes para este indicador.")
            
            st.markdown("---")
    
    # Rodapé
    st.caption(f"Dados atualizados até {periodos[-1] if periodos else 'data desconhecida'}. Análise gerada em {datetime.now().strftime('%d/%m/%Y')}.")
    st.caption("Fonte: Sistema de Informação em Saúde para a Atenção Básica (SISAB)")

def main():
    # Carregar dados
    municipios_dados = carregar_dados_municipios()
    
    # Barra lateral
    st.sidebar.title("Previne Brasil")
    
    # Tentar carregar logo
    logo_path = "logo.png"
    try:
        if Path(logo_path).exists():
            st.sidebar.image(logo_path, use_container_width=True)
        else:
            st.sidebar.write("Previne Brasil")
    except Exception as e:
        logger.error(f"Erro ao carregar logo: {str(e)}")
    
    if not municipios_dados:
        st.error("Nenhum dado de município encontrado! Verifique os arquivos JSON.")
        return
    
    # Seleção de município
    municipios = list(municipios_dados.keys())
    municipio_selecionado = st.sidebar.selectbox("Selecione o Município:", municipios)
    
    # Informações sobre o programa
    st.sidebar.markdown("---")
    st.sidebar.subheader("Sobre o Previne Brasil")
    st.sidebar.info("""
    O Programa Previne Brasil foi instituído pela Portaria nº 2.979/2019, 
    e estabelece novo modelo de financiamento da Atenção Primária à Saúde.
    
    Indicadores avaliados:
    - Pré-natal
    - Saúde da mulher
    - Imunização infantil
    - Doenças crônicas
    """)
    
    # Mostrar dashboard
    if municipio_selecionado:
        dados_municipio = municipios_dados[municipio_selecionado]
        gerar_dashboard(municipio_selecionado, dados_municipio)

if __name__ == "__main__":
    main()
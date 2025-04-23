"""
Processador de Dados SISAB - Sistema de Informação em Saúde para a Atenção Básica

Este programa realiza o processamento e consolidação de dados dos indicadores de saúde do SISAB.
Funcionalidades:
- Processa arquivos CSV contendo indicadores de desempenho do SISAB
- Extrai informações de cabeçalho (indicador, estado, municípios)
- Organiza e consolida os dados por município
- Salva dados processados em arquivos JSON individuais para cada município
- Gera um relatório consolidado com estatísticas dos indicadores por município

Os arquivos processados devem estar na mesma pasta do script com prefixo 'paine-indicador' e extensão '.csv'.
Os resultados são salvos na mesma pasta onde o script está sendo executado.
"""

import os
import csv
import json
import pandas as pd
import re
from collections import defaultdict
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='desempenho.log'
)

logger = logging.getLogger('sisab_consolidado')

def extrair_info_cabecalho(linhas):
    """Extrai informações do cabeçalho do arquivo CSV."""
    info = {}
    
    for linha in linhas:
        if not linha:  # Pula linhas vazias
            continue
            
        linha_texto = linha[0] if isinstance(linha, list) and len(linha) > 0 else linha
        
        if isinstance(linha_texto, str):
            if 'Indicador:' in linha_texto:
                info['indicador'] = linha_texto.replace('Indicador:', '').strip().replace('"', '')
            elif 'Estado:' in linha_texto:
                info['estado'] = linha_texto.replace('Estado:', '').strip()
            elif 'Município:' in linha_texto:
                # Remove aspas e extrai nomes dos municípios
                municipios_texto = linha_texto.replace('Município:', '').replace('"', '').strip()
                info['municipios'] = [m.strip() for m in municipios_texto.split(',')]
    
    return info

def processar_arquivo_csv(caminho_arquivo):
    """Processa um único arquivo CSV do SISAB."""
    dados_processados = []
    indicador = None
    
    try:
        # Tentar diferentes encodings comuns no Brasil
        encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
        conteudo = None
        encoding_usado = None
        
        for encoding in encodings:
            try:
                with open(caminho_arquivo, 'r', encoding=encoding) as f:
                    conteudo = f.readlines()
                encoding_usado = encoding
                logger.info(f"Arquivo lido com encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if not conteudo:
            logger.error(f"Não foi possível decodificar o arquivo {caminho_arquivo}")
            return [], None
        
        # Extrair informações do cabeçalho
        info_cabecalho = extrair_info_cabecalho(conteudo[:10])
        logger.info(f"Informações do cabeçalho: {info_cabecalho}")
        indicador = info_cabecalho.get('indicador', 'Não especificado')
        
        # Procurar linha que contém o cabeçalho das colunas (UF;IBGE;Município;...)
        indice_cabecalho = None
        for i, linha in enumerate(conteudo):
            if 'UF;IBGE;' in linha:
                indice_cabecalho = i
                break
        
        if indice_cabecalho is None:
            logger.error(f"Cabeçalho não encontrado no arquivo {caminho_arquivo}")
            return [], indicador
        
        # Extrair cabeçalho das colunas
        cabecalho = conteudo[indice_cabecalho].strip().split(';')
        
        # Processar linhas de dados (após o cabeçalho)
        for i in range(indice_cabecalho + 1, len(conteudo)):
            linha = conteudo[i].strip()
            
            # Ignorar linhas vazias ou linhas de rodapé
            if not linha or 'Fonte:' in linha:
                continue
                
            # Dividir a linha em campos
            campos = linha.split(';')
            if len(campos) >= 3:  # Precisa ter pelo menos UF, IBGE e Município
                try:
                    uf = campos[0].strip()
                    ibge = campos[1].strip()
                    municipio = campos[2].strip()
                    
                    # Validação básica
                    if not municipio or len(municipio) < 2:
                        continue
                    
                    # Criar estrutura de dados
                    dados = {
                        'UF': uf,
                        'IBGE': ibge,
                        'Municipio': municipio,
                        'Indicador': indicador
                    }
                    
                    # Adicionar valores para cada período (que começa na posição 3)
                    for j in range(3, min(len(cabecalho), len(campos))):
                        if j < len(cabecalho) and cabecalho[j]:
                            periodo = cabecalho[j].strip()
                            if j < len(campos) and campos[j]:
                                valor = campos[j].strip()
                                if valor and valor != ';':
                                    dados[periodo] = valor
                    
                    dados_processados.append(dados)
                    logger.info(f"Processados dados de: {municipio}")
                except Exception as e:
                    logger.error(f"Erro ao processar linha {i}: {str(e)}")
        
        return dados_processados, indicador
    except Exception as e:
        logger.error(f"Erro ao processar arquivo {caminho_arquivo}: {str(e)}")
        return [], indicador

def processar_arquivos_csv():
    """Processa todos os arquivos CSV da pasta atual."""
    diretorio = os.path.dirname(__file__)  # Usa a pasta atual onde o script está
    dados_por_municipio = defaultdict(list)
    indicadores_processados = set()
    
    logger.info(f"Buscando arquivos CSV em: {diretorio}")
    
    # Verificando se o diretório existe
    if not os.path.exists(diretorio):
        logger.error(f"Diretório não encontrado: {diretorio}")
        return dados_por_municipio
    
    # Listando arquivos com padrão específico
    arquivos = [f for f in os.listdir(diretorio) if f.startswith('paine-indicador') and f.endswith('.csv')]
    logger.info(f"Encontrados {len(arquivos)} arquivos para processar")
    
    for arquivo in arquivos:
        logger.info(f"Processando arquivo: {arquivo}")
        caminho_arquivo = os.path.join(diretorio, arquivo)
        
        dados_processados, indicador = processar_arquivo_csv(caminho_arquivo)
        if indicador:
            indicadores_processados.add(indicador)
        
        # Agrupar dados por município
        for dados in dados_processados:
            municipio = dados['Municipio']
            dados['Fonte'] = arquivo  # Adicionar nome do arquivo de origem
            dados_por_municipio[municipio].append(dados)
    
    logger.info(f"Total de indicadores processados: {len(indicadores_processados)}")
    logger.info(f"Total de municípios encontrados: {len(dados_por_municipio)}")
    
    return dados_por_municipio

def ordenar_indicadores(dados):
    """
    Ordena os indicadores conforme a sequência especificada e adiciona numeração.
    
    Ordem:
    1. Pré-natal
    2. Sífilis/HIV
    3. Gestantes saúde bucal
    4. Citopatológico
    5. Vacina
    6. Hipertensão
    7. Diabetes
    """
    # Definir padrões para identificar cada categoria
    categorias = [
        (1, ["pré-natal", "pre-natal", "consultas", "12ª", "12a", "semana", "gestação"]),
        (2, ["sífilis", "sifilis", "hiv", "exames"]),
        (3, ["odontológico", "odontologico", "bucal"]),
        (4, ["citopatológico", "citopatologico"]),
        (5, ["criança", "crianca", "vacina", "difteria", "tétano", "tetano", "coqueluche", "hepatite", "poliomielite"]),
        (6, ["hipertensão", "hipertensao", "pressão", "pressao", "arterial"]),
        (7, ["diabetes", "glicada", "hemoglobina"])
    ]
    
    def atribuir_categoria(indicador):
        # Converte para minúsculas para facilitar a comparação
        indicador_lower = indicador.lower()
        
        for num, palavras_chave in categorias:
            for palavra in palavras_chave:
                if palavra in indicador_lower:
                    return num
        
        # Se não encontrou, retorna um valor alto para ficar no final
        return 99
    
    dados_ordenados = []
    for item in dados:
        ordem = atribuir_categoria(item.get('Indicador', ''))
        # Adiciona o número do indicador e mantém descrição original
        item['Indicador'] = f"{ordem}. {item.get('Indicador', '')}"
        # Adiciona a ordem como uma propriedade separada para facilitar a ordenação
        item['Ordem'] = ordem
        dados_ordenados.append(item)
    
    # Ordena os dados pela propriedade 'Ordem'
    dados_ordenados.sort(key=lambda x: x['Ordem'])
    
    # Remove a propriedade temporária de ordenação
    for item in dados_ordenados:
        item.pop('Ordem', None)
    
    return dados_ordenados

def salvar_dados_por_municipio(dados_por_municipio):
    """Salva os dados consolidados em arquivos JSON separados por município na pasta atual."""
    diretorio_saida = os.path.dirname(__file__)  # Usa a pasta atual onde o script está
    
    for municipio, dados in dados_por_municipio.items():
        try:
            # Ordenar os indicadores conforme a sequência especificada
            dados_ordenados = ordenar_indicadores(dados)
            
            # Sanitizar nome do município para usar como nome de arquivo
            nome_arquivo = re.sub(r'[\\/*?:"<>|]', "", municipio)
            caminho_arquivo = os.path.join(diretorio_saida, f"{nome_arquivo}.json")
            
            with open(caminho_arquivo, 'w', encoding='utf-8') as f:
                json.dump(dados_ordenados, f, ensure_ascii=False, indent=4)
                
            logger.info(f"Dados do município {municipio} salvos em: {caminho_arquivo}")
        except Exception as e:
            logger.error(f"Erro ao salvar dados do município {municipio}: {str(e)}")

def gerar_relatorio_consolidado(dados_por_municipio):
    """Gera um relatório consolidado em formato texto na pasta atual."""
    diretorio_saida = os.path.dirname(__file__)  # Usa a pasta atual onde o script está
    caminho_relatorio = os.path.join(diretorio_saida, "relatorio_consolidado.txt")
    
    try:
        with open(caminho_relatorio, 'w', encoding='utf-8') as f:
            f.write("RELATÓRIO CONSOLIDADO DOS INDICADORES SISAB\n")
            f.write("=========================================\n\n")
            
            f.write(f"Data de geração: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}\n")
            f.write(f"Total de municípios processados: {len(dados_por_municipio)}\n\n")
            
            # Agrupar indicadores para estatísticas
            todos_indicadores = set()
            todos_periodos = set()
            
            for municipio, dados in dados_por_municipio.items():
                for item in dados:
                    todos_indicadores.add(item.get('Indicador', 'Não especificado'))
                    for chave in item.keys():
                        # Identificar colunas que são períodos (ex: 2022 Q1 (%))
                        if re.match(r'^\d{4}\s+Q\d+\s+\(%\)$', chave):
                            todos_periodos.add(chave)
            
            f.write(f"Total de indicadores: {len(todos_indicadores)}\n")
            f.write(f"Períodos analisados: {', '.join(sorted(todos_periodos))}\n\n")
            
            # Resumo por município
            for municipio, dados in dados_por_municipio.items():
                f.write(f"Município: {municipio}\n")
                f.write("-" * len(f"Município: {municipio}") + "\n")
                
                # Agrupar dados por indicador
                indicadores_por_municipio = {}
                for item in dados:
                    indicador = item.get('Indicador', 'Não especificado')
                    if indicador not in indicadores_por_municipio:
                        indicadores_por_municipio[indicador] = 0
                    indicadores_por_municipio[indicador] += 1
                
                f.write(f"Total de indicadores: {len(indicadores_por_municipio)}\n")
                
                # Listar indicadores disponíveis
                for ind, count in indicadores_por_municipio.items():
                    f.write(f"  - {ind}: {count} registros\n")
                
                f.write("\n")
        
        logger.info(f"Relatório consolidado gerado em: {caminho_relatorio}")
        return caminho_relatorio
    except Exception as e:
        logger.error(f"Erro ao gerar relatório consolidado: {str(e)}")
        return None

if __name__ == "__main__":
    logger.info("Iniciando processamento dos arquivos SISAB")
    
    try:
        # Processar arquivos CSV
        dados_por_municipio = processar_arquivos_csv()
        
        # Salvar dados por município
        salvar_dados_por_municipio(dados_por_municipio)
        
        # Gerar relatório consolidado
        caminho_relatorio = gerar_relatorio_consolidado(dados_por_municipio)
        
        # Informar resultado
        total_municipios = len(dados_por_municipio)
        total_registros = sum(len(dados) for dados in dados_por_municipio.values())
        
        print(f"Processamento concluído!")
        print(f"Total de municípios processados: {total_municipios}")
        print(f"Total de registros processados: {total_registros}")
        print(f"Os dados foram salvos na pasta atual: {os.path.dirname(__file__)}")
        if caminho_relatorio:
            print(f"Relatório consolidado gerado em: {caminho_relatorio}")
        
        logger.info(f"Processamento concluído com sucesso: {total_municipios} municípios, {total_registros} registros")
    except Exception as e:
        logger.error(f"Erro durante o processamento: {str(e)}")
        print(f"Ocorreu um erro durante o processamento. Consulte o arquivo de log para detalhes.")
# 📸 Analisador Forense de Imagens

Um software avançado para detecção de manipulação e compressão múltipla em imagens usando técnicas forenses digitais.

## 🎯 Funcionalidades

### 🔍 Análises Implementadas

| Análise | Descrição | Técnica |
|---------|-----------|---------|
| **Lei de Benford** | Verifica a distribuição natural dos primeiros dígitos | Processamento Digital |
| **Ruído do Sensor** | Analisa padrões de ruído para detectar inconsistências | Análise Física |
| **Compressão Múltipla** | Detecta se a imagem foi salva várias vezes | Teoria da Informação |
| **Detecção de Clonagem** | Identifica regiões copiadas/coladas | Visão Computacional |
| **Análise de Iluminação** | Verifica consistência na direção da luz | Óptica Física |
| **Detecção de Resampling** | Detecta redimensionamentos | Processamento de Sinais |
| **Metadados EXIF** | Analisa inconsistências nos dados da câmera | Metadados |

## 🚀 Instalação

### Pré-requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### 📦 Instalação das Dependências

```bash
# Instalar todas as dependências
pip install opencv-python numpy scipy scikit-learn Pillow matplotlib exifread

# Ou instalar uma por uma
pip install opencv-python
pip install numpy
pip install scipy
pip install scikit-learn
pip install Pillow
pip install matplotlib
pip install exifread
🛠️ Instalação no Windows
cmd

# Abra o Prompt de Comando como Administrador
python -m pip install --upgrade pip
pip install opencv-python numpy scipy scikit-learn Pillow matplotlib exifread

🐧 Instalação no Linux/Mac
bash

# Atualizar pip e instalar dependências
python3 -m pip install --upgrade pip
pip3 install opencv-python numpy scipy scikit-learn Pillow matplotlib exifread

💻 Como Usar
Método 1: Execução Direta
python

# Salve o código como 'analisador_forense.py' e execute:
python analisador_forense.py

Método 2: Uso como Módulo
python

from analisador_forense import DetectorManipulacaoAvancado

# Inicializar o detector
detector = DetectorManipulacaoAvancado()

# Analisar uma imagem
resultados = detector.analise_completa("caminho/para/sua/imagem.jpg")

Método 3: Análise Individual
python

detector = DetectorManipulacaoAvancado()

# Análises específicas
resultado_benford = detector.analisar_lei_benford("imagem.jpg")
resultado_clonagem = detector.detectar_clonagem("imagem.jpg")
resultado_compressao = detector.detectar_compressao_multipla("imagem.jpg")

📊 Interpretação dos Resultados
🟢 Resultados Normais

    Correlação Benford > 0.95

    Inconsistência de Iluminação < 30°

    Score Compressão < 1.5

    Poucos ou nenhum clone detectado

🟡 Resultados Suspeitos

    Correlação Benford: 0.90-0.95

    Inconsistência de Iluminação: 30°-45°

    Score Compressão: 1.5-2.0

    Alguns clones detectados

🔴 Resultados de Manipulação

    Correlação Benford < 0.90

    Inconsistência de Iluminação > 45°

    Score Compressão > 2.0

    Múltiplos clones detectados

    Metadados inconsistentes

🧠 Metodologias Científicas
Lei de Benford
python

# Imagens naturais seguem a distribuição:
P(d) = log10(1 + 1/d) para d = 1,2,...,9
# Onde P(d) é a probabilidade do dígito d ser o primeiro

Análise de Ruído

    Calcula a variância do ruído residual

    Verifica consistência entre quadrantes

    Detecta suavização artificial

Detecção de Clonagem

    Divide imagem em blocos

    Calcula similaridade entre blocos

    Usa correlação e características estatísticas


🐛 Solução de Problemas

Análise muito lenta

    O código inclui otimizações automáticas

    Imagens grandes são redimensionadas

    Use imagens com menos de 10MB para melhor performance

🔬 Exemplo de Saída
======================================================================
ANÁLISE FORENSE COMPLETA DE IMAGEM
======================================================================

🔍 Lei de Benford (DCT):
   ✅ NORMAL - Sem indícios de manipulação
   correlacao: 0.9723
   distancia_euclidiana: 0.0456
   total_amostras: 267245

🔍 Detecção de Clonagem:
   ✅ NORMAL - Sem indícios de manipulação
   clones_detectados: 2
   total_blocos_analisados: 180

🔍 Compressão Múltipla:
   ✅ NORMAL - Sem indícios de manipulação
   entropia_histograma: 7.5474
   score_compressao_multipla: 1.2033

======================================================================
RESULTADO FINAL: IMAGEM PROVAVELMENTE AUTÊNTICA (14.3%)
======================================================================

📝 Formatos Suportados

    JPEG (.jpg, .jpeg)

    PNG (.png)

    BMP (.bmp)

    TIFF (.tiff, .tif)

    WebP (.webp)

⚠️ Limitações

    Eficácia reduzida em imagens muito comprimidas

    Pode gerar falsos positivos em condições de iluminação complexas

    Análise de metadados depende das informações incluídas pela câmera
